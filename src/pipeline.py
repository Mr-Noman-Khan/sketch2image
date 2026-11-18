"""Load Stable Diffusion 1.5 + ControlNet and generate images from sketches."""
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNETS = {
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
}
DEFAULT_NEGATIVE = "blurry, low quality, deformed, distorted, watermark, text"


def load_pipeline(control_type="scribble", device="cuda"):
    """Load the base Stable Diffusion model with the chosen ControlNet."""
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNETS[control_type], torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, torch_dtype=torch.float16
    )
    # UniPC gives good results in ~20-30 steps (faster than the default scheduler)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    return pipe


def generate(
    pipe,
    control_image,
    prompt,
    negative_prompt=DEFAULT_NEGATIVE,
    steps=25,
    guidance_scale=7.5,
    controlnet_scale=1.0,
    seed=42,
    num_images=1,
):
    """Generate image(s) that follow the sketch's structure and the prompt."""
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=control_image,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
        num_images_per_prompt=num_images,
    )
    return result.images
