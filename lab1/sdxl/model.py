"""Model wrapper for serving Diffuser SDXL model."""

import argparse
import typing
import torch
from io import BytesIO
from base64 import b64encode
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)

def read_image(image):
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64

class Model:
    """Wrapper for SDXL interrogator model."""

    def __init__(self):
        """Initialize the model."""
        print("Initializing SDXL pipeline...")
        # Init the VAE
        self._vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        # Init the SDXL Pipeline
        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=self._vae,
            torch_dtype=torch.float16,
        )
        # Enable memory optimization
        self._pipe.unet.to(memory_format=torch.channels_last)
        # Send to GPU
        self._pipe.to("cuda")
        # Enable xformers optimization
        self._pipe.enable_xformers_memory_efficient_attention()
        # Enable Scheduler to user Euler Ancestral
        self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._pipe.scheduler.config
        )
        print("SDXL pipeline has been initialized!")

    def predict(self, inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        """Return interrogation for the given image.

        :param inputs: dict of inputs containing model inputs
               with the following keys:
        - "prompt" (mandatory): The SDXL prompt (string).
        - "negative_prompt" (optional): The SDXL negative prompt (string).
        - "guidance_scale" (optional): The SDXL guidance scale (float).
        - "num_inference_steps" (optional): The SDXL number of steps (int).
        - "width" (optional): The width of the image (int).
        - "height" (optional): The height of the image (int).
        - "seed" (optional): The seed of the image (int).

        :return: a dict containing these keys:

        - "image": A base64-encoded image.
        """
        print("Got a new request for SDXL...")

        # Read the various SDXL pipeline parameters from the input dictionary
        prompt = inputs.get("prompt", "")
        negative_prompt = inputs.get("negative_prompt", "")
        guidance_scale = inputs.get("guidance_scale", 7.5)
        num_inference_steps = inputs.get("num_inference_steps", 20)
        width = inputs.get("width", 1024)
        height = inputs.get("height", 1024)
        seed = inputs.get("seed", -1)

        print("\tPrompt: {}".format(prompt))
        print("\tNegative prompt: {}".format(negative_prompt))
        print("\tGuidance scale: {}".format(guidance_scale))
        print("\tNum inference steps: {}".format(num_inference_steps))
        print("\tWidth: {}".format(width))
        print("\tHeight: {}".format(height))
        print("\tSeed: {}".format(seed))

        # Initialize the generator's random seed
        generator = torch.Generator(device="cuda").manual_seed(seed)
        # Run the SDXL pipeline
        images = self._pipe(
            prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
        ).images

        print("SDXL request completed!")

        # Prepare the response dictionary
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
