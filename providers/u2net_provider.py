# providers/u2net_provider.py
import os
import cv2
import numpy as np
import onnxruntime as ort

class U2NetProvider:
    _session = None
    _input_name = None

    @classmethod
    def load_model(cls, variant="default"):
        """
        Loads the U²‑Net ONNX model.
        Optionally, choose a variant (e.g., "human").
        """
        if cls._session is None:
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            if variant == "default":
                model_path = os.path.join(base_dir, "u2net.onnx")
            else:
                model_path = os.path.join(base_dir, f"u2net_{variant}.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"U2NET model not found at {model_path}")

            cls._session = ort.InferenceSession(model_path)
            inputs = cls._session.get_inputs()
            if not inputs:
                raise ValueError("No inputs found in the U2NET model.")
            cls._input_name = inputs[0].name
            print("U2NET loaded. Model input name:", cls._input_name)
        return cls._session

    @classmethod
    def preprocess(cls, image, target_size=(320, 320)):
        """
        Prepares the image for U²‑Net:
         - If image has an alpha channel, convert from RGBA → BGR.
         - Resize to the target size.
         - Normalize pixel values to [0,1].
         - Convert from HWC to CHW and add a batch dimension.
        Returns:
         (input_tensor, (orig_w, orig_h))
        """
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        orig_h, orig_w, _ = image.shape
        resized = cv2.resize(image, target_size)
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (C, H, W)
        img = np.expand_dims(img, 0)  # (1, C, H, W)
        return img, (orig_w, orig_h)

    @classmethod
    def refine_mask(cls, prob_map, original_size, threshold=0.05):
        """
        1. Resize to original size.
        2. Threshold at 0.05 → convert [0,1] to [0,255].
        3. Bilateral filter to preserve edges while smoothing noise.
        4. Re-threshold to ensure binary mask.
        5. Feather with a small Gaussian blur.
        6. Normalize to [0..255].
        """
        # 1. Resize to original size
        prob_map = cv2.resize(prob_map, original_size, interpolation=cv2.INTER_LINEAR)

        # 2. Threshold to create a binary mask in [0,1]
        _, binary = cv2.threshold(prob_map, threshold, 1.0, cv2.THRESH_BINARY)
        mask = (binary * 255).astype(np.uint8)

        # 3. Bilateral filter: smooth noise while preserving edges
        # (d=9, sigmaColor=75, sigmaSpace=75 are typical defaults)
        mask_bf = cv2.bilateralFilter(mask.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

        # 4. Re-threshold to ensure the result is binary again
        _, mask = cv2.threshold(mask_bf, 128, 255, cv2.THRESH_BINARY)

        # 5. Feather edges with a small Gaussian blur
        # Convert to float for a smoother blur, then we'll normalize
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)

        # 6. Normalize final mask to [0..255] and convert back to uint8
        mask_f = cv2.normalize(mask_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return mask_f

    @classmethod
    def postprocess(cls, prediction, original_size, threshold=0.05):
        """
        Handles output shapes (1,1,H,W) or (1,H,W), then calls refine_mask.
        """
        if len(prediction.shape) == 4 and prediction.shape[1] == 1:
            prob_map = prediction[0, 0, :, :]
        elif len(prediction.shape) == 3:
            prob_map = prediction[0]
        else:
            raise ValueError(f"Unexpected output shape: {prediction.shape}")

        return cls.refine_mask(prob_map, original_size, threshold)

    @classmethod
    def get_salient_mask(cls, image, threshold=0.05):
        """
        Main entry point for generating the mask:
         - Load U²‑Net if needed.
         - Preprocess the image.
         - Run inference.
         - Postprocess with bilateral filter + feathering.
         - Return a binary mask [H,W] in uint8 with 0/255.
        """
        session = cls.load_model()
        input_tensor, original_size = cls.preprocess(image)
        outputs = session.run(None, {cls._input_name: input_tensor})
        mask = cls.postprocess(outputs[0], original_size, threshold)
        return mask
