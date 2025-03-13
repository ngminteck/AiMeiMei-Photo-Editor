import os
import cv2
import numpy as np
import onnxruntime as ort

class U2NetProvider:
    _session = None
    _input_name = None

    # Default configuration parameters (can be updated via set_config)
    _target_size = (320, 320)
    _bilateral_d = 9
    _bilateral_sigmaColor = 75
    _bilateral_sigmaSpace = 75
    _gaussian_kernel_size = 5  # Must be odd

    @classmethod
    def load_model(cls, variant="default"):
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
    def set_config(cls, target_size=None, bilateral_d=None, bilateral_sigmaColor=None,
                   bilateral_sigmaSpace=None, gaussian_kernel_size=None):
        if target_size is not None:
            cls._target_size = target_size
        if bilateral_d is not None:
            cls._bilateral_d = bilateral_d
        if bilateral_sigmaColor is not None:
            cls._bilateral_sigmaColor = bilateral_sigmaColor
        if bilateral_sigmaSpace is not None:
            cls._bilateral_sigmaSpace = bilateral_sigmaSpace
        if gaussian_kernel_size is not None:
            cls._gaussian_kernel_size = gaussian_kernel_size
        print("U2NET configuration updated.")

    @classmethod
    def preprocess(cls, image, target_size=None):
        if target_size is None:
            target_size = cls._target_size
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
        # Resize to original size.
        prob_map = cv2.resize(prob_map, original_size, interpolation=cv2.INTER_LINEAR)
        # Threshold to create binary mask in [0,1].
        _, binary = cv2.threshold(prob_map, threshold, 1.0, cv2.THRESH_BINARY)
        mask = (binary * 255).astype(np.uint8)
        # Apply bilateral filter with current config.
        mask_bf = cv2.bilateralFilter(mask.astype(np.float32), d=cls._bilateral_d,
                                        sigmaColor=cls._bilateral_sigmaColor,
                                        sigmaSpace=cls._bilateral_sigmaSpace)
        # Re-threshold.
        _, mask = cv2.threshold(mask_bf, 128, 255, cv2.THRESH_BINARY)
        # Feather edges with a Gaussian blur.
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (cls._gaussian_kernel_size, cls._gaussian_kernel_size), 0)
        mask_f = cv2.normalize(mask_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mask_f

    @classmethod
    def postprocess(cls, prediction, original_size, threshold=0.05):
        if len(prediction.shape) == 4 and prediction.shape[1] == 1:
            prob_map = prediction[0, 0, :, :]
        elif len(prediction.shape) == 3:
            prob_map = prediction[0]
        else:
            raise ValueError(f"Unexpected output shape: {prediction.shape}")
        return cls.refine_mask(prob_map, original_size, threshold)

    @classmethod
    def get_salient_mask(cls, image, threshold=0.05):
        session = cls.load_model()
        input_tensor, original_size = cls.preprocess(image)
        outputs = session.run(None, {cls._input_name: input_tensor})
        mask = cls.postprocess(outputs[0], original_size, threshold)
        return mask
