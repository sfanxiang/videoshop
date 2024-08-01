import argparse
import os
import sys
import time

sys.path = ["."] + sys.path

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--strength", type=float, default=0.55)
parser.add_argument("--min-guidance-scale", type=float, default=4.0)
parser.add_argument("--max-guidance-scale", type=float, default=4.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

import imageio.v3 as iio
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from transformers import CLIPVisionModelWithProjection

from data_module import PreprocessedDataset
from euler_discrete_inverse import EulerDiscreteInverseScheduler
from model_utils import controlled_generation, controlled_generation_inverse, vae_decode, vae_encode

pl.seed_everything(args.seed)


class ItemDataset:
    def __init__(self, video, image):
        self.video = video
        self.image = image

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        assert idx in [0, 1]

        video = self.video
        image = self.image if idx == 1 else self.video[0]

        return {
            "video": video,
            "image1": image,
            "image2": image,
        }


scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid", subfolder="scheduler")
inverse_scheduler = EulerDiscreteInverseScheduler.from_config(scheduler.config)
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", torch_dtype=torch.float16
).cuda()
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder", torch_dtype=torch.float16
).cuda()
unet_video = UNetSpatioTemporalConditionModel.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", subfolder="unet", torch_dtype=torch.float16
).cuda()

video = iio.imread(args.video)
video = torch.tensor(video)[..., :3]
image = iio.imread(args.image)
image = torch.tensor(image)[..., :3]

dataset = PreprocessedDataset(ItemDataset(video, image), 576, 1024, 224, 224, resize=True)

pl.seed_everything(args.seed)

item = dataset[0]
video = item["video"]
image1 = item["image1"]
image2 = item["image2"]

video = video[None].cuda().half()
image1 = image1[None].cuda().half()
image2 = image2[None].cuda().half()

time_start = time.time()

latents = vae_encode(vae, video)
mean = latents.reshape(latents.size(1), -1).mean(dim=1)
std = latents.reshape(latents.size(1), -1).std(dim=1)
latents = latents - mean[None, :, None, None, None]
latents = latents / (std[None, :, None, None, None] + 1e-6)

noise_aug_strength = 0.2

image_noise = torch.randn(image2.size(), dtype=image2.dtype, device=image2.device)
image2 = image2 + noise_aug_strength * image_noise
with torch.no_grad():
    image_latents = vae_encode(vae, image2[:, None], random=False, scale=False)
mean = image_latents.reshape(image_latents.size(1), -1).mean(dim=1)
std = image_latents.reshape(image_latents.size(1), -1).std(dim=1)
image_latents = image_latents - mean[None, :, None, None, None]
image_latents = image_latents / (std[None, :, None, None, None] + 1e-6)
image_latents = image_latents / vae.config.scaling_factor

strength = args.strength

outputs = controlled_generation_inverse(
    scheduler=inverse_scheduler,
    vae=vae,
    image_encoder=image_encoder,
    unet_video=unet_video,
    video=video,
    image1=image1,
    image2=image2,
    num_inference_steps=25,
    fps=7,
    motion_bucket_id=127,
    noise_aug_strength=noise_aug_strength,
    latents=latents,
    image_latents=image_latents,
    strength=strength,
    min_guidance_scale=1.0,
    max_guidance_scale=1.0,
)
noisy_latents = outputs["latents"]

item2 = dataset[1]
image1 = item2["image1"]
image2 = item2["image2"]
image1 = image1[None].cuda().half()
image2 = image2[None].cuda().half()

noise_aug_strength = 0.2
image_latents = None

outputs = controlled_generation(
    scheduler=scheduler,
    vae=vae,
    image_encoder=image_encoder,
    unet_video=unet_video,
    video=video,
    image1=image1,
    image2=image2,
    num_inference_steps=25,
    fps=7,
    motion_bucket_id=127,
    noise_aug_strength=noise_aug_strength,
    noisy_latents=noisy_latents,
    image_latents=image_latents,
    strength=strength,
    min_guidance_scale=args.min_guidance_scale,
    max_guidance_scale=args.max_guidance_scale,
)
output_latents = outputs["latents"]

with torch.no_grad():
    image_latents = vae_encode(vae, image2[:, None], random=False, scale=True)

output_latents = output_latents - output_latents[:, :1].flatten(-2, -1).mean(dim=-1)[..., None, None]
output_latents = (
    output_latents
    / output_latents[:, :1].flatten(-2, -1).std(dim=-1)[..., None, None]
    * image_latents.flatten(-2, -1).std(dim=-1)[..., None, None]
)
output_latents = output_latents + image_latents.flatten(-2, -1).mean(dim=-1)[..., None, None]

print(time.time() - time_start)

with torch.no_grad():
    generated_video = vae_decode(vae, output_latents)

iio.imwrite(args.output, ((generated_video[0] + 1.0) / 2.0).clamp(0.0, 1.0).numpy(force=True), fps=7)
