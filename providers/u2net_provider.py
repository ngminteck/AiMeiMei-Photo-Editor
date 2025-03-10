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
        Loads the U²-Net ONNX model.
        Optionally, you can choose a different variant (e.g., "human") if available.
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
        Preprocesses the image:
         - If the image has an alpha channel, remove it (convert RGBA → BGR).
         - Resize to the target size.
         - Normalize pixel values to the range [0, 1].
         - Rearrange dimensions to CHW and add a batch dimension.
        Returns the input tensor and the original size as (orig_w, orig_h).
        """
        if image.shape[2] == 4:
            # Remove alpha channel if present.
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        orig_h, orig_w, _ = image.shape
        resized = cv2.resize(image, target_size)
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # (C, H, W)
        img = np.expand_dims(img, 0)   # (1, C, H, W)
        return img, (orig_w, orig_h)

    @classmethod
    def refine_mask(cls, mask, original_size):
        """
        Refines the segmentation mask:
         - Resizes the probability map to the original image size.
         - Converts probabilities (0 to 1) to an 8-bit scale (0 to 255).
         - Applies Gaussian blur (kernel 7×7) to soften the edges.
         - Uses erosion (iterations=2) followed by dilation (iterations=2)
           with a 3×3 kernel to remove halos and fill small gaps.
         - Normalizes the result.
        """
        # Resize the mask to the original size using linear interpolation.
        mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_LINEAR)
        # Convert from [0,1] to [0,255]
        mask = (mask * 255).astype(np.uint8)
        # Soften edges with Gaussian blur.
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        # Define a 3x3 kernel.
        kernel = np.ones((3, 3), np.uint8)
        # Apply erosion to remove unwanted halos.
        mask = cv2.erode(mask, kernel, iterations=2)
        # Then dilation to restore the object size.
        mask = cv2.dilate(mask, kernel, iterations=1)
        # Normalize the mask.
        mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
        return mask

    @classmethod
    def postprocess(cls, prediction, original_size):
        """
        Postprocesses the model's output:
         - Handles outputs of shape (1, 1, H, W) or (1, H, W).
         - Calls refine_mask() to produce a smooth, 8-bit segmentation mask.
        """
        # Handle different output shapes.
        if len(prediction.shape) == 4 and prediction.shape[1] == 1:
            prob_map = prediction[0, 0, :, :]
        elif len(prediction.shape) == 3:
            prob_map = prediction[0]
        else:
            raise ValueError(f"Unexpected output shape: {prediction.shape}")

        refined_mask = cls.refine_mask(prob_map, original_size)
        return refined_mask

    @classmethod
    def get_salient_mask(cls, image, variant="default"):
        """
        Runs U²-Net segmentation on an image and returns a refined binary mask.
         - Loads the model (optionally a different variant).
         - Preprocesses the image.
         - Runs inference.
         - Postprocesses the output using the refinement pipeline.
        """
        session = cls.load_model(variant)
        input_tensor, original_size = cls.preprocess(image)
        outputs = session.run(None, {cls._input_name: input_tensor})
        mask = cls.postprocess(outputs[0], original_size)
        return mask
