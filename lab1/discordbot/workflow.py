import requests
import discord
import uuid
import os

from random import randint
from PIL import Image
from io import BytesIO
from base64 import b64encode, b64decode
from octoai.client import Client

client = Client()

SDXL_ENDPOINT_URL = os.environ["SDXL_ENDPOINT_URL"]

# A helper function that reads a PIL Image objects and returns a base 64 encoded string
def encode_image(image: Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64


# A helper function that reads a base64 encoded string and returns a PIL Image object
def decode_image(image_str: str) -> Image:
    return Image.open(BytesIO(b64decode(image_str)))


# Let's define the OctoShop as a self-contained function
def octoshop(image: Image, user_prompt: str, user_style: dict) -> (Image, str, str):
    # OctoAI endpoint URLs
    clip_endpoint_url = "https://dockercon23-clip-4jkxk521l3v1.octoai.run"
    llama2_endpoint_url = (
        "https://dockercon23-llama2-4jkxk521l3v1.octoai.run/v1/chat/completions"
    )

    # STEP 1
    # Feed that image into CLIP interrogator
    clip_request = {
        "mode": "fast",
        "image": encode_image(image),
    }
    output = client.infer(
        endpoint_url="{}/predict".format(clip_endpoint_url), inputs=clip_request
    )
    clip_labels = output["completion"]["labels"]
    clip_labels = clip_labels.split(",")[0]

    # STEP 2
    # Feed that CLIP label and the user prompt into a LLAMA model
    llama_prompt = "\
    ### Instruction: In a single sentence, {}: {}\
    ### Response:".format(
        user_prompt, clip_labels
    )
    llama_inputs = {
        "model": "llama-2-7b-chat",
        "messages": [
            {
                "role": "system",
                "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
            },
            {"role": "user", "content": "{}".format(llama_prompt)},
        ],
        "stream": False,
        "max_tokens": 256,
    }
    outputs = client.infer(endpoint_url=llama2_endpoint_url, inputs=llama_inputs)
    llama2_text = outputs.get("choices")[0].get("message").get("content")

    # STEP 3
    # Feed the LLAMA2 text into the SDXL model
    SDXL_payload = {
        "prompt": user_style["prompt"].replace("{prompt}", llama2_text),
        "negative_prompt": user_style["negative_prompt"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": randint(1, 1000000),
    }
    # Run inference on the OctoAI SDXL model container running locally
    output = client.infer(
        endpoint_url="{}/predict".format(SDXL_ENDPOINT_URL), inputs=SDXL_payload
    )
    image_string = output["completion"]["image"]

    return image_string, clip_labels, llama2_text


def octoshop_workflow(image_url: str):
    # Process encode the input image into a string
    r = requests.get(image_url)
    image = Image.open(BytesIO(r.content))

    # Set the user prompt
    user_prompt = "set in outer space"

    # Set the style of SDXL
    user_style = {
        "name": "sai-digital art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    }

    image_string, clip_labels, llama2_output = octoshop(image, user_prompt, user_style)
    buffer = BytesIO(b64decode(image_string))
    filename = f"{str(uuid.uuid4())}.jpg"
    image_file = discord.File(fp=buffer, filename=filename)

    return image_file, clip_labels, llama2_output
