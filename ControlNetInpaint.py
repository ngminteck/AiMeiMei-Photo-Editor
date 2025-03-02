import torch
from torchvision import transforms
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionInpaintPipeline
import glob
import os
import shutil

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

    pipe = StableDiffusionInpaintPipeline.from_pretrained(stable_diffusion_dir, torch_dtype=torch_dtype,
                                                          local_files_only=True).to(device, dtype=torch_dtype)
    controlnet = ControlNetModel.from_pretrained(controlnet_dir, local_files_only=True).to(device, dtype=torch_dtype)

    return pipe

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGBA")
    original_size = image.size

    alpha = image.split()[3]
    mask = alpha.point(lambda p: 255 if p < 128 else 0).convert("L")

    image = image.convert("RGB")
    return image, mask, original_size

def make_divisible_by_8(size):
    """Ensure both height and width are divisible by 8."""
    width, height = size
    width = (width // 8) * 8
    height = (height // 8) * 8
    return width, height

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_controlnet()

    test_image_path = "images/segmentation_result/2_people_together/2_people_together_background.png"
    test_image, test_mask, original_size = preprocess_image(test_image_path)

    adjusted_size = make_divisible_by_8(original_size)

    reference_images_dir = "images/reference_images"
    os.makedirs(reference_images_dir, exist_ok=True)
    reference_images = [Image.open(img).convert("RGB") for img in glob.glob(f"{reference_images_dir}/*.*")]

    transform = transforms.ToTensor()

    location = "Merlion Park"
    country = "Singapore"
    prompt = (
        f"This photo is taken at {location}, {country}. "
        f"Restore missing areas by accurately extending the visible surroundings. "
        f"Use reference images to ensure color and texture consistency. "
        f"Preserve landmarks, lighting, and depth perception."
    )

    result = pipe(
        prompt=prompt,
        image=test_image.resize(adjusted_size, Image.Resampling.LANCZOS),
        mask_image=test_mask.resize(adjusted_size, Image.Resampling.LANCZOS),
        conditioning_image=[img.resize(adjusted_size, Image.Resampling.LANCZOS) for img in reference_images] if reference_images else None,
        height=adjusted_size[1],
        width=adjusted_size[0]
    ).images[0]

    result = result.resize(original_size, Image.Resampling.LANCZOS)

    output_dir = "images/controlnet_result"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(test_image_path))
    result.save(output_path)
