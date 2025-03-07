#providers/controlnet_model_provider.py
import os
import torch
import shutil
import glob
from diffusers import ControlNetModel, StableDiffusionInpaintPipeline
from torchvision import transforms
from PIL import Image

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def clean_huggingface_cache(model_path):
    """Remove unnecessary Hugging Face cache directories and .lock files."""
    for root, dirs, files in os.walk(model_path, topdown=False):
        for name in files:
            if name.endswith(".lock"):
                os.remove(os.path.join(root, name))
        for name in dirs:
            if name.startswith("models--") or name == "temp":
                shutil.rmtree(os.path.join(root, name), ignore_errors=True)

def get_latest_snapshot(model_path):
    """Find and move the correct snapshot folder for a downloaded model."""
    if os.path.exists(model_path):
        for subdir in os.listdir(model_path):
            snapshot_path = os.path.join(model_path, subdir, "snapshots")
            if os.path.exists(snapshot_path):
                snapshots = sorted(os.listdir(snapshot_path), reverse=True)
                if snapshots:
                    latest_snapshot = os.path.join(snapshot_path, snapshots[0])
                    for file_name in os.listdir(latest_snapshot):
                        src = os.path.join(latest_snapshot, file_name)
                        dest = os.path.join(model_path, file_name)
                        if not os.path.exists(dest):
                            shutil.move(src, dest)
                    shutil.rmtree(os.path.dirname(latest_snapshot), ignore_errors=True)
                    return model_path
    return model_path

def check_and_download_model(model_name, model_path, is_controlnet=False):
    """Check if the model exists; if not, download and move it to the correct directory."""
    if is_controlnet:
        model_path = os.path.join(model_path, "controlnet")
    else:
        model_path = os.path.join(model_path, "stable-diffusion")

    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"{model_name} already exists. Skipping download.")
        return

    print(f"{model_name} not found. Downloading...")
    temp_dir = os.path.join("models", "temp")

    if is_controlnet:
        ControlNetModel.from_pretrained(model_name, cache_dir=temp_dir)
    else:
        StableDiffusionInpaintPipeline.from_pretrained(model_name, cache_dir=temp_dir)

    correct_model_path = get_latest_snapshot(temp_dir)
    os.makedirs(model_path, exist_ok=True)
    for file_name in os.listdir(correct_model_path):
        src = os.path.join(correct_model_path, file_name)
        dest = os.path.join(model_path, file_name)
        if not os.path.exists(dest):
            shutil.move(src, dest)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"{model_name} downloaded and saved in {model_path}")

def load_controlnet():
    """
    Downloads (if necessary) and loads the Stable Diffusion Inpaint Pipeline.
    Note: In a full integration you might want to integrate the ControlNet model
    directly into the pipeline. For now we return the inpainting pipeline as per the prototype.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    controlnet_dir = os.path.join(models_dir, "controlnet")
    stable_diffusion_dir = os.path.join(models_dir, "stable-diffusion")
    os.makedirs(controlnet_dir, exist_ok=True)
    os.makedirs(stable_diffusion_dir, exist_ok=True)

    check_and_download_model("stabilityai/stable-diffusion-2-inpainting", models_dir, is_controlnet=False)
    check_and_download_model("lllyasviel/control_v11p_sd15_inpaint", models_dir, is_controlnet=True)

    clean_huggingface_cache(models_dir)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        stable_diffusion_dir,
        torch_dtype=torch_dtype,
        local_files_only=True
    ).to(device, dtype=torch_dtype)
    return pipe

def make_divisible_by_8(size):
    """Ensure both width and height are divisible by 8."""
    width, height = size
    width = (width // 8) * 8
    height = (height // 8) * 8
    return (width, height)
