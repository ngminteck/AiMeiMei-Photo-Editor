# providers/realesrgan_provider.py
import os
import cv2
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class RealESRGANProvider:
    _instance = None

    @classmethod
    def load_model(cls, model_variant="x4plus", half=True, device=None):
        if cls._instance is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            if model_variant == "x4plus":
                model_path = os.path.join(base_dir, "RealESRGAN_x4plus.pth")
            else:
                raise ValueError("Unsupported model variant: " + model_variant)
            # Initialize the RRDBNet model architecture (parameters as required by RealESRGAN_x4plus)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
            cls._instance = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                pre_pad=10,
                half=half,
                device=device
            )
        return cls._instance

    @classmethod
    def upscale(cls, image, model_variant="x4plus"):
        """
        Upscales the input image (in BGR format) using RealESRGAN.
        Returns the upscaled image in BGR format.
        """
        sr_model = cls.load_model(model_variant=model_variant)
        # Convert image from BGR to RGB as expected by the model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            output, _ = sr_model.enhance(image_rgb)
        except Exception as e:
            raise RuntimeError("RealESRGAN processing failed: " + str(e))
        # Convert the output from RGB back to BGR
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output_bgr
