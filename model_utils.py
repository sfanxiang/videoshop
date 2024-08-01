import math

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn


def video_to_images(x):
    shape = x.shape[:2]
    y = x.flatten(0, 1)
    return y, shape


def images_to_video(x, shape):
    return x.unflatten(0, shape)


def controlled_unet_forward(
    unet_video,
    video_latents,
    timestep,
    encoder_hidden_states,
    added_time_ids,
):
    sample = unet_video(
        video_latents,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_time_ids=added_time_ids,
    ).sample
    return sample


def _get_add_time_ids(
    unet,
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    device,
    batch_size,
    num_videos_per_prompt,
    do_classifier_free_guidance,
):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

    if do_classifier_free_guidance:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])

    return add_time_ids


def vae_encode(vae, video, generator=None, random=False, scale=True):
    # video: BTHWC
    output = []

    step = 14 if video.size(1) <= 14 else 10
    for i in range(0, video.size(1), step):
        v = video[:, i : i + step]
        pixel_values, video_shape = video_to_images(v)  # (BT)HWC
        pixel_values = pixel_values.permute(0, 3, 1, 2)  # (BT)CHW
        latent_dist = vae.encode(pixel_values).latent_dist
        if random:
            latents = latent_dist.sample(generator)
        else:
            latents = latent_dist.mode()
        if scale:
            latents = latents * vae.config.scaling_factor
        latents = images_to_video(latents, video_shape)  # BTCHW
        output.append(latents)
    return torch.cat(output, dim=1)


def vae_decode(vae, latents):
    # latents: BTCHW
    latents = 1 / vae.config.scaling_factor * latents
    num_frames = latents.size(1)
    latents, video_shape = video_to_images(latents)  # (BT)CHW
    video = vae.decode(latents, num_frames=num_frames).sample  # (BT)CHW
    video = images_to_video(video, video_shape)  # BTCHW
    video = video.permute(0, 1, 3, 4, 2)  # BTHWC
    return video


def controlled_generation(
    scheduler,
    vae,
    image_encoder,
    unet_video,
    video,
    image1,
    image2,
    generator=None,
    vae_random=False,
    fps=7,
    motion_bucket_id=127,
    num_inference_steps=25,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    first_frame_guidance_scale=None,
    noise_aug_strength=0.02,
    strength=1.0,
    noisy_latents=None,
    image_latents=None,
):
    do_classifier_free_guidance = max_guidance_scale > 1.0

    _, video_shape = video_to_images(video)
    with torch.no_grad():
        latents = vae_encode(vae, video, generator=generator, random=vae_random)  # BTCHW

    input_noisy_latents = noisy_latents
    noise = torch.randn(latents.size(), dtype=latents.dtype, device=latents.device, generator=generator)
    noisy_latents = noise

    encoder_hidden_states = image_encoder(image1.permute(0, 3, 1, 2)).image_embeds[:, None]

    if do_classifier_free_guidance:
        negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
        encoder_hidden_states = torch.cat([negative_encoder_hidden_states, encoder_hidden_states])

    if image_latents is None:
        image_noise = torch.randn(image2.size(), dtype=image2.dtype, device=image2.device, generator=generator)
        image2 = image2 + noise_aug_strength * image_noise
        with torch.no_grad():
            image_latents = vae_encode(vae, image2[:, None], generator=generator, random=vae_random, scale=False)
    image_latents = image_latents.expand(-1, video_shape[1], -1, -1, -1)  # BTCHW

    if do_classifier_free_guidance:
        negative_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([negative_image_latents, image_latents])

    add_time_ids = _get_add_time_ids(
        unet_video,
        fps=fps - 1,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        dtype=noisy_latents.dtype,
        device=noisy_latents.device,
        batch_size=video_shape[0],
        num_videos_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    scheduler.set_timesteps(num_inference_steps, device=noisy_latents.device)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]
    num_inference_steps = num_inference_steps - t_start

    noisy_latents = noisy_latents * scheduler.init_noise_sigma

    if input_noisy_latents is not None:
        noisy_latents = input_noisy_latents

    guidance_scale = torch.linspace(
        min_guidance_scale, max_guidance_scale, video_shape[1], device=noisy_latents.device, dtype=noisy_latents.dtype
    ).unsqueeze(0)
    if first_frame_guidance_scale is not None:
        guidance_scale[:, 0] = first_frame_guidance_scale
    guidance_scale = guidance_scale.repeat(video_shape[0], 1)
    guidance_scale = guidance_scale[:, :, None, None, None]

    for i, t in enumerate(timesteps):
        latent_model_input = noisy_latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

        with torch.no_grad():
            unet_output = controlled_unet_forward(
                unet_video=unet_video,
                video_latents=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=add_time_ids,
            )

        # perform guidance
        if do_classifier_free_guidance:
            unet_output_uncond, unet_output_cond = unet_output.chunk(2)
            unet_output = unet_output_uncond + guidance_scale * (unet_output_cond - unet_output_uncond)

        noisy_latents = scheduler.step(unet_output, t, noisy_latents).prev_sample

    with torch.no_grad():
        video = vae_decode(vae, noisy_latents)

    outputs = {}
    outputs["generated_video"] = video
    outputs["latents"] = noisy_latents
    return outputs


def controlled_generation_inverse(
    scheduler,
    vae,
    image_encoder,
    unet_video,
    video,
    image1,
    image2,
    generator=None,
    vae_random=False,
    fps=7,
    motion_bucket_id=127,
    num_inference_steps=25,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    noise_aug_strength=0.02,
    strength=1.0,
    latents=None,
    image_latents=None,
):
    do_classifier_free_guidance = max_guidance_scale > 1.0

    _, video_shape = video_to_images(video)
    if latents is None:
        with torch.no_grad():
            latents = vae_encode(vae, video, generator=generator, random=vae_random)  # BTCHW

    noisy_latents = latents.clone()
    noise = torch.randn(latents.size(), dtype=latents.dtype, device=latents.device, generator=generator)

    encoder_hidden_states = image_encoder(image1.permute(0, 3, 1, 2)).image_embeds[:, None]

    if do_classifier_free_guidance:
        negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states)
        encoder_hidden_states = torch.cat([negative_encoder_hidden_states, encoder_hidden_states])

    if image_latents is None:
        image_noise = torch.randn(image2.size(), dtype=image2.dtype, device=image2.device, generator=generator)
        image2 = image2 + noise_aug_strength * image_noise
        with torch.no_grad():
            image_latents = vae_encode(vae, image2[:, None], generator=generator, random=vae_random, scale=False)
    image_latents = image_latents.expand(-1, video_shape[1], -1, -1, -1)  # BTCHW

    if do_classifier_free_guidance:
        negative_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([negative_image_latents, image_latents])

    add_time_ids = _get_add_time_ids(
        unet_video,
        fps=fps - 1,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        dtype=noisy_latents.dtype,
        device=noisy_latents.device,
        batch_size=video_shape[0],
        num_videos_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    scheduler.set_timesteps(num_inference_steps, device=noisy_latents.device)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]
    num_inference_steps = num_inference_steps - t_start

    noisy_latents = noisy_latents * scheduler.init_noise_sigma

    guidance_scale = torch.linspace(
        min_guidance_scale, max_guidance_scale, video_shape[1], device=noisy_latents.device, dtype=noisy_latents.dtype
    ).unsqueeze(0)
    guidance_scale = guidance_scale.repeat(video_shape[0], 1)
    guidance_scale = guidance_scale[:, :, None, None, None]

    scheduler._init_step_index(timesteps[-1])

    iter_steps = list(enumerate(timesteps))
    iter_steps.reverse()
    for i, t in iter_steps:
        sigma_next = (
            scheduler.sigmas[scheduler.step_index + 2] if len(scheduler.sigmas) > scheduler.step_index + 2 else None
        )
        sigma_current = scheduler.sigmas[scheduler.step_index + 1]
        sigma_prev = scheduler.sigmas[scheduler.step_index]

        old_noisy_latents = noisy_latents

        if sigma_current.item() > 1:
            noisy_latents = (noisy_latents - latents) / sigma_current * sigma_prev
            noisy_latents = latents + noisy_latents
        else:
            noisy_latents = noise * sigma_prev * latents.std()
            noisy_latents = latents + noisy_latents

        latent_model_input = noisy_latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

        with torch.no_grad():
            unet_output = controlled_unet_forward(
                unet_video=unet_video,
                video_latents=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=add_time_ids,
            )

        noisy_latents = old_noisy_latents

        # perform guidance
        if do_classifier_free_guidance:
            unet_output_uncond, unet_output_cond = unet_output.chunk(2)
            unet_output = unet_output_uncond + guidance_scale * (unet_output_cond - unet_output_uncond)

        noisy_latents = scheduler.step(unet_output, t, noisy_latents).prev_sample

    outputs = {}
    outputs["latents"] = noisy_latents
    return outputs
