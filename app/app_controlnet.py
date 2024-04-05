#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
import sys

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import gradio as gr
from diffusion.model.utils import prepare_prompt_ar, resize_and_crop_tensor

import numpy as np
import torch
from PIL import Image as PILImage
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import _log_api_usage_once, make_grid, save_image
from diffusion.model.t5 import T5Embedder
from diffusers import PixArtAlphaPipeline
from diffusion import DPMS, SASolverSampler
from diffusion.data.datasets import *
from diffusion.model.hed import HEDdetector
from diffusion.model.nets import PixArt_XL_2, PixArtMS_XL_2, ControlPixArtHalf, ControlPixArtMSHalf
from diffusion.model.utils import resize_and_crop_tensor
from diffusion.utils.misc import read_config
from tools.download import find_model
from PIL import Image


DESCRIPTION = """![Logo](https://raw.githubusercontent.com/PixArt-alpha/PixArt-alpha.github.io/master/static/images/logo.png)
        # PixArt-Delta (ControlNet)
        #### [PixArt-Alpha 1024px](https://github.com/PixArt-alpha/PixArt-alpha) is a transformer-based text-to-image diffusion system trained on text embeddings from T5. 
        #### This demo uses the [PixArt-alpha/PixArt-XL-2-1024-ControlNet](https://huggingface.co/PixArt-alpha/PixArt-ControlNet/tree/main) checkpoint.
        #### This demo uses the [PixArt-alpha/PixArt-XL-2-512-ControlNet](https://huggingface.co/PixArt-alpha/PixArt-ControlNet/tree/main) checkpoint.
        #### English prompts ONLY; 提示词仅限英文
        ### <span style='color: red;'>Please use the image size corresponding to the model as input to get the best performance. (eg. 1024px for PixArt-XL-2-1024-ControlNet.pth)
        """
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU �� This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "1") == "1"
PORT = int(os.getenv("DEMO_PORT", "15432"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def ndarr_image(tensor: Union[torch.Tensor, List[torch.Tensor]], **kwargs, ) -> None:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr


style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]


styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"
SCHEDULE_NAME = ["DPM-Solver", "SA-Solver"]
DEFAULT_SCHEDULE_NAME = "DPM-Solver"

def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative


def save_image(img):
    unique_name = str(uuid.uuid4()) + '.png'
    save_path = os.path.join(f'output/online_demo_img/{datetime.now().date()}')
    os.makedirs(save_path, exist_ok=True)
    unique_name = os.path.join(save_path, unique_name)
    img.save(unique_name)
    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@torch.inference_mode()
def generate(
        prompt: str,
        given_image = None,
        negative_prompt: str = "",
        style: str = DEFAULT_STYLE_NAME,
        use_negative_prompt: bool = False,
        seed: int = 0,
        width: int = 1024,
        height: int = 1024,
        schedule: str = 'DPM-Solver',
        dpms_guidance_scale: float = 4.5,
        sas_guidance_scale: float = 3,
        dpms_inference_steps: int = 14,
        sas_inference_steps: int = 25,
        randomize_seed: bool = False,
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    strength = 0.5
    c_vis = given_image

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask\
        = pipe.encode_prompt(prompt=prompt, negative_prompt=negative_prompt)
    prompt_embeds, negative_prompt_embeds = prompt_embeds[:, None], negative_prompt_embeds[:, None]

    prompt_clean, prompt_show, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device)  # ar for aspect ratio
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    caption_embs, emb_masks = llm_embed_model.get_text_embeddings(prompts)
    caption_embs = caption_embs[:, None]

    torch.cuda.empty_cache()

    # condition process
    if given_image is not None:
        ar = torch.tensor([given_image.size[1] / given_image.size[0]], device=device)[None]
        custom_hw = torch.tensor([given_image.size[1], given_image.size[0]], device=device)[None]
        closest_hw = base_ratios[min(base_ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))]
        hw = torch.tensor(closest_hw, device=device)[None]
        condition_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(int(min(closest_hw))),
            T.CenterCrop([int(closest_hw[0]), int(closest_hw[1])]),
            T.ToTensor(),
        ])

        given_image = condition_transform(given_image).unsqueeze(0).to(device)
        hed_edge = hed(given_image) * strength
        hed_edge = TF.normalize(hed_edge, [.5], [.5])
        hed_edge = hed_edge.repeat(1, 3, 1, 1).to(weight_dtype)
        posterior = vae.encode(hed_edge).latent_dist
        condition = posterior.sample()
        c = condition * config.scale_factor
        c_vis = vae.decode(condition)['sample']
        c_vis = torch.clamp(127.5 * c_vis + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    else:
        c = None
        ar = torch.tensor([int(height) / int(width)], device=device)[None]
        custom_hw = torch.tensor([int(height), int(width)], device=device)[None]
        closest_hw = base_ratios[min(base_ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))]
        hw = torch.tensor(closest_hw, device=device)[None]

    latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)

    # Sample images:
    if schedule == 'DPM-Solver':
        # Create sampling noise:
        n = len(prompts)
        z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=prompt_attention_mask, c=c)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                          condition=caption_embs,
                          uncondition=negative_prompt_embeds,
                          cfg_scale=dpms_guidance_scale,
                          model_kwargs=model_kwargs)
        samples = dpm_solver.sample(
            z,
            steps=dpms_inference_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        ).to(weight_dtype)
    elif schedule == "SA-Solver":
        # Create sampling noise:
        n = len(prompts)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=prompt_attention_mask, c=c)
        sas_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sas_solver.sample(
            S=sas_inference_steps,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=negative_prompt_embeds,
            unconditional_guidance_scale=sas_guidance_scale,
            model_kwargs=model_kwargs,
        )[0].to(weight_dtype)

    samples = vae.decode(samples / config.scale_factor).sample
    torch.cuda.empty_cache()
    samples = resize_and_crop_tensor(samples, custom_hw[0, 1], custom_hw[0, 0])
    samples = PILImage.fromarray(ndarr_image(samples, normalize=True, value_range=(-1, 1)))
    image_paths = [save_image(samples)]
    c_vis = PILImage.fromarray(c_vis) if c_vis is not None else samples
    c_paths = [save_image(c_vis)]
    print(image_paths)
    print(c_paths)
    return image_paths, c_paths, seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config")
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--model_path', type=str)
    return parser.parse_args()


args = get_args()
config = read_config(args.config)
device = "cuda" if torch.cuda.is_available() else "cpu"

assert args.image_size in [512, 1024], "We only provide pre-trained models for 512x512 and 1024x1024 resolutions."
lewei_scale = {512: 1, 1024: 2}
latent_size = args.image_size // 8
weight_dtype = torch.float16
print(f"Inference with {weight_dtype}")

if torch.cuda.is_available():
    hed = HEDdetector(False).to(device)
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        transformer=None,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pipe.to(device)
    print("Loaded on Device!")
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    llm_embed_model = T5Embedder(device=device, local_cache=True, cache_dir='data/t5_ckpts', torch_dtype=torch.float)
    assert args.image_size == config.image_size
    if config.image_size == 512:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[config.image_size])
        print('model architecture ControlPixArtHalf and image size is 512')
        model = ControlPixArtHalf(model).to(device)
    elif config.image_size == 1024:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[config.image_size])
        print('model architecture ControlPixArtMSHalf and image size is 1024')
        model = ControlPixArtMSHalf(model).to(device)

    state_dict = find_model(args.model_path)['state_dict']
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    elif 'base_model.pos_embed' in state_dict:
        del state_dict['base_model.pos_embed']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('Missing keys (missing pos_embed is normal): ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

if __name__ == "__main__":
    generate(
        prompt="Leather sofa in a rich caramel colour, coffee table with a natural wood top and intricate cut-out white metal base, large flat-screen television mounted on a dark green circular feature wall, wooden media console with a dark brown finish, ceiling fan with a light wood finish and frosted glass light",
        given_image=Image.open("asset/images/controlnet/living_room_base.jpeg"),
        negative_prompt="",
        style="(No style)",
        use_negative_prompt=False,
        seed=0,
        width=1024,
        height=1024,
        schedule="SA-Solver",
        dpms_guidance_scale=7.0,
        sas_guidance_scale=7,
        dpms_inference_steps=30,
        sas_inference_steps=60,
        randomize_seed=True,
    )
