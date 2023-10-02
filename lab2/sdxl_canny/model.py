"""Model wrapper for serving Diffuser SDXL+Canny model."""

import argparse
import typing
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
from diffusers import (
    ControlNetModel, 
    StableDiffusionXLControlNetPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler,
)

def read_image(image):
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64

class Model:
    """Wrapper for SDXL+Canny interrogator model."""

    def __init__(self):
        """Initialize the model."""
        print("Initializing SDXL+Canny pipeline...")

        # TODO: add the controlnet
        # controlnet =

        # TODO: add the VAE
        # vae = ...

        # TODO: add the SDXL+Controlnet Pipeline
        # self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(...)

        # Begin SDXL optimizations

        # Enable memory optimization
        self._pipe.unet.to(memory_format=torch.channels_last)
        # Send to GPU
        self._pipe.to('cuda')
        # Enable xformers optimization
        self._pipe.enable_xformers_memory_efficient_attention()
        # Enable Scheduler to user Euler Ancestral
        self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._pipe.scheduler.config
        )

        # End SDXL optimizations
        print("SDXL+Canny pipeline has been initialized!")

    def predict(self, inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        """Return interrogation for the given image.

        :param inputs: dict of inputs containing model inputs
               with the following keys:
        - "image" (mandatory): A base64-encoded image (string).
        - "prompt" (optional): The SDXL prompt (string).
        - "negative_prompt" (optional): The SDXL negative prompt (string).
        - "guidance_scale" (optional): The SDXL guidance scale (float).
        - "seed" (optional): The seed of the image (int).
        - "controlnet_conditioning_scale" (optional): The Controlnet conditioning scale (float).
        - "control_guidance_start" (optional): The Controlnet guidance start (float).
        - "control_guidance_end" (optional): The Controlnet guidance end (float).

        :return: a dict containing these keys:

        - "image": A base64-encoded image.
        """
        print("Got a new request for SDXL+Canny...")

        # Read the various SDXL pipeline parameters from the input dictionary
        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "")
        guidance_scale = inputs.get("guidance_scale", 7.5)
        num_inference_steps = inputs.get("num_inference_steps", 20)
        seed = inputs.get("seed", -1)
        controlnet_conditioning_scale = inputs.get("controlnet_conditioning_scale", 0.5)
        control_guidance_start = inputs.get("control_guidance_start", 0)
        control_guidance_end = inputs.get("control_guidance_end", 0.67)
        image = inputs.get("image", None)

        print("\tPrompt: {}".format(prompt))
        print("\tNegative prompt: {}".format(negative_prompt))
        print("\tGuidance scale: {}".format(guidance_scale))
        print("\tNum inference steps: {}".format(num_inference_steps))
        print("\tControlNet Conditioning scale: {}".format(controlnet_conditioning_scale))
        print("\tControlNet Guidance start: {}".format(control_guidance_start))
        print("\tControlNet Guidance end: {}".format(control_guidance_end))

        # Convert input image string base64 into Pillow Image
        image_bytes = BytesIO(b64decode(image))
        image_pil = Image.open(image_bytes)
        # Derive the dimensions of the rescaled image
        width, height = image_pil.size
        print("\tWidth: {}".format(width))
        print("\tHeight: {}".format(height))

        # Initialize the generator's random seed
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # TODO: add canny filter

        # TODO: run SDXL pipeline
        # images = self._pipe().images

        print("SDXL+Canny request completed!")

        # Return first image
        response = {"completion": {"image": read_image(images[0])}}

        return response

    @classmethod
    def fetch(cls) -> None:
        """Pre-fetches the model for implicit caching by Transfomers."""
        # Running the constructor is enough to fetch this model.
        cls()

def main():

    """Entry point for interacting with this model via CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    if args.fetch:
        Model.fetch()

if __name__ == "__main__":
    main()
