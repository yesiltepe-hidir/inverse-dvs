import argparse
import math
import os
from typing import Any, Dict, List, Optional, TypedDict, Union

import torch
import random
import numpy as np
import torchvision.transforms as T
from diffusers.models.autoencoders import AutoencoderKLCogVideoX
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, DDIMInverseScheduler
from diffusers.utils import export_to_video

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error.
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort: skip

from diffusers.utils import load_video
from huggingface_hub import hf_hub_download

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DDIMInversionArguments(TypedDict):
    model_path: str
    prompt: str
    video_path: str
    output_path: str
    guidance_scale: float
    num_inference_steps: int
    skip_frames_start: int
    skip_frames_end: int
    frame_sample_step: Optional[int]
    max_num_frames: int
    width: int
    height: int
    fps: int
    dtype: torch.dtype
    seed: int
    device: torch.device


def get_args() -> DDIMInversionArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subfolder", type=str, required=True, help="Subfolder of the video"
    )
    parser.add_argument(
        "--model_path", type=str, required=False, default="THUDM/CogVideoX-5b-I2V", help="Path of the pretrained model"
    )
    parser.add_argument(
        "--lora_path", type=str, required=False, default="I2V5B_final_i38800_nearest_lora_weights.safetensors", help="Path of the lora weights"
    )
    parser.add_argument(
        "--prompt", type=str, required=False, default="", help="Prompt for the direct sample procedure"
    )
    parser.add_argument(
        "--video_path", type=str, required=False, default="davis_data/data_copy/03_reading_cat.mp4", help="Path of the video for inversion"
    )
    parser.add_argument(
        "--output_path", type=str, default="inversions", help="Path of the output videos"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=30, help="Number of inference steps"
    )
    parser.add_argument(
        "--skip_frames_start", type=int, default=0, help="Number of skipped frames from the start"
    )
    parser.add_argument(
        "--skip_frames_end", type=int, default=0, help="Number of skipped frames from the end"
    )
    parser.add_argument(
        "--frame_sample_step", type=int, default=None, help="Temporal stride of the sampled frames"
    )
    parser.add_argument(
        "--max_num_frames", type=int, default=81, help="Max number of sampled frames"
    )
    parser.add_argument("--width", type=int, default=720, help="Resized width of the video frames")
    parser.add_argument(
        "--height", type=int, default=480, help="Resized height of the video frames"
    )
    parser.add_argument("--fps", type=int, default=30, help="Frame rate of the output videos")
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Dtype of the model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )

    args = parser.parse_args()
    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.device = torch.device(args.device)

    return DDIMInversionArguments(**vars(args))


def get_video_frames(
    video_path: str,
    width: int,
    height: int,
    skip_frames_start: int,
    skip_frames_end: int,
    max_num_frames: int,
    frame_sample_step: Optional[int],
) -> torch.FloatTensor:
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
        video_num_frames = len(video_reader)
        start_frame = min(skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - skip_frames_end)

        if end_frame <= start_frame:
            indices = [start_frame]
        elif end_frame - start_frame <= max_num_frames:
            indices = list(range(start_frame, end_frame))
        else:
            step = frame_sample_step or (end_frame - start_frame) // max_num_frames
            indices = list(range(start_frame, end_frame, step))

        frames = video_reader.get_batch(indices=indices)
        frames = frames[:max_num_frames].float()  # ensure that we don't go over the limit

        # Choose first (4k + 1) frames as this is how many is required by the VAE
        selected_num_frames = frames.size(0)
        remainder = (3 + selected_num_frames) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        assert frames.size(0) % 4 == 1

        # Normalize the frames
        transform = T.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        frames = torch.stack(tuple(map(transform, frames)), dim=0)

        return frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]


def encode_video_frames(
    vae: AutoencoderKLCogVideoX, video_frames: torch.FloatTensor, generator: Optional[torch.Generator] = None
) -> torch.FloatTensor:
    video_frames = video_frames.to(device=vae.device, dtype=vae.dtype)
    video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    # latent_dist = vae.encode(x=video_frames).latent_dist.sample().transpose(1, 2)
    latent_dist = vae.encode(x=video_frames).latent_dist.mode().transpose(1, 2)
    latent_dist = latent_dist * vae.config.scaling_factor
    return latent_dist


def export_latents_to_video(
    pipeline: CogVideoXImageToVideoPipeline, latents: torch.FloatTensor, video_path: str, fps: int
):
    print('latents: ', latents.shape)
    video = pipeline.decode_latents(latents)
    frames = pipeline.video_processor.postprocess_video(video=video, output_type="pil")
    export_to_video(video_frames=frames[0], output_video_path=video_path, fps=fps)


# Modified from CogVideoXImageToVideoPipeline.__call__
def sample(
    pipeline: CogVideoXImageToVideoPipeline,
    latents: torch.FloatTensor,
    image_latents: torch.FloatTensor,
    scheduler: Union[DDIMInverseScheduler, CogVideoXDDIMScheduler],
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6,
    use_dynamic_cfg: bool = False,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    reference_latents: torch.FloatTensor = None,
) -> torch.FloatTensor:
    pipeline._guidance_scale = guidance_scale
    pipeline._attention_kwargs = attention_kwargs
    pipeline._interrupt = False

    device = pipeline._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt,
        negative_prompt,
        do_classifier_free_guidance,
        device=device,
    )
    if do_classifier_free_guidance:
        print("negative prompt is used...")
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    if reference_latents is not None:
        prompt_embeds = torch.cat([prompt_embeds] * 2, dim=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)
    pipeline._num_timesteps = len(timesteps)

    # 5. Prepare latents.
    latents = latents.to(device=device) * scheduler.init_noise_sigma

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    if isinstance(scheduler, DDIMInverseScheduler):  # Inverse scheduler does not accept extra kwargs
        extra_step_kwargs = {}

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        pipeline._prepare_rotary_positional_embeddings(
            height=latents.size(3) * pipeline.vae_scale_factor_spatial,
            width=latents.size(4) * pipeline.vae_scale_factor_spatial,
            num_frames=latents.size(1),
            device=device,
        )
        if pipeline.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)

    trajectory = torch.zeros_like(latents).unsqueeze(0).repeat(len(timesteps), 1, 1, 1, 1, 1)
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline.interrupt:
                continue

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

            if reference_latents is not None:
                reference = reference_latents[i]
                reference = torch.cat([reference] * 2) if do_classifier_free_guidance else reference
                reference = torch.cat([reference, latent_image_input], dim=2)
                latent_model_input = torch.cat([latent_model_input, reference], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = pipeline.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            if reference_latents is not None:  # Recover the original batch size
                noise_pred, _ = noise_pred.chunk(2)

            # perform guidance
            if use_dynamic_cfg:
                pipeline._guidance_scale = 1 + guidance_scale * (
                    (
                        1
                        - math.cos(
                            math.pi
                            * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0
                        )
                    )
                    / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipeline.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the noisy sample x_t-1 -> x_t
            # FIXIT: Problem is here
            latents = scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
            )[0]
            latents = latents.to(prompt_embeds.dtype)
            trajectory[i] = latents

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
            ):
                progress_bar.update()

    # Offload all models
    pipeline.maybe_free_model_hooks()

    return trajectory


@torch.no_grad()
def ddim_inversion(
    model_path: str,
    prompt: str,
    video_path: str,
    output_path: str,
    guidance_scale: float,
    num_inference_steps: int,
    skip_frames_start: int,
    skip_frames_end: int,
    frame_sample_step: Optional[int],
    max_num_frames: int,
    width: int,
    height: int,
    fps: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
    lora_path: str = None,
    subfolder: str = None,
):
    # set seed
    set_seed(seed)

    video_name = video_path.split('/')[-2]
    print(f"Processing: {subfolder}_latents_{video_name}_{num_inference_steps}")

    # Load the model
    pipeline: CogVideoXImageToVideoPipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device=device)
    
    if not pipeline.transformer.config.use_rotary_positional_embeddings:
        raise NotImplementedError("This script supports CogVideoX 5B model only.")

    print('Not loading LoRA weights: ', lora_path)
    if lora_path != "None":
        print('Loading LoRA weights from: ', lora_path)
        base_url = 'Eyeline-Research/Go-with-the-Flow'
        lora_path = hf_hub_download(repo_id=base_url, filename=lora_path)
        pipeline.load_lora_weights(lora_path)
    
    print('output_path: ', output_path)
    image = load_video(video_path)[0]
    image = pipeline.video_processor.preprocess(image, height=480, width=720).to(device, dtype=torch.bfloat16)

    video_frames = get_video_frames(
        video_path=video_path,
        width=width,
        height=height,
        skip_frames_start=skip_frames_start,
        skip_frames_end=skip_frames_end,
        max_num_frames=49,
        frame_sample_step=frame_sample_step,
    ).to(device=device)
    video_latents = encode_video_frames(vae=pipeline.vae, video_frames=video_frames)
    
    _, image_latents = pipeline.prepare_latents(
        image=image,
        batch_size=1,
        num_channels_latents=16,
        num_frames=49,
        height=480,
        width=720,
        dtype=torch.bfloat16,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    
    # TODO: Change here later --------
    inverse_scheduler = DDIMInverseScheduler(**pipeline.scheduler.config)
    print("--------------------------------")
    inverse_scheduler.config['prediction_type'] = 'sample'
    print('inverse_scheduler: ', inverse_scheduler.config['prediction_type'])
    print("--------------------------------")

    inverse_latents = sample(
        pipeline=pipeline,
        latents=video_latents,
        image_latents=image_latents,
        scheduler=inverse_scheduler,
        prompt="",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    inverse_latents = reversed(inverse_latents)
    torch.save(inverse_latents, os.path.join(output_path, f"{subfolder}_latents_{video_name}_{num_inference_steps}.pt"))
    export_latents_to_video(pipeline=pipeline, latents=inverse_latents[0], video_path=os.path.join(output_path, f"{subfolder}_inverse_render_{video_name}_{num_inference_steps}.mp4"), fps=fps)

if __name__ == "__main__":
    arguments = get_args()
    ddim_inversion(**arguments)