import argparse
import math
import os
from typing import Any, Dict, List, Optional, TypedDict, Union

import torch
import random
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.models.autoencoders import AutoencoderKLCogVideoX
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps

from pipelines.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.schedulers import DDIMInverseScheduler
from diffusers.utils import export_to_video
from models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error.
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort: skip

from diffusers.utils import load_video
from huggingface_hub import hf_hub_download
import numpy as np
import rp
from scipy.ndimage import zoom
from schedulers.scheduling_ddim_cogvideox import CogVideoXDDIMScheduler
from utils.grid import create_video_grid
from utils.downsize_mask import downsize_mask

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
    treshold_idx: int


def get_args() -> DDIMInversionArguments:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path of the pretrained model"
    )

    parser.add_argument(
        "--k_order", type=int, required=False, help="Order of the k-diffusion"
    )

    parser.add_argument(
        "--lora_path", type=str, required=False, help="Path of the lora weights"
    )
    parser.add_argument(
        "--inverted_latent_path", type=str, required=False, help="Path of the inverted latents"
    )
    
    parser.add_argument(
        "--mask_path", type=str, required=False, help="Path of the mask"
    )
    parser.add_argument(
        "--depth_path", type=str, required=False, help="Path of the depth"
    )

    parser.add_argument(
        "--treshold_idx", type=int, required=False, default=-5, help="Index to use for the threshold"
    )

    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt for the direct sample procedure"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path of the video for inversion"
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="Path of the output videos"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--preservation_scale", type=float, default=3.0, help="Preservation scale"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--width", type=int, default=720, help="Resized width of the video frames"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Resized height of the video frames"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frame rate of the output videos"
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Dtype of the model"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for the random number generator"
    )

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
    vae: AutoencoderKLCogVideoX, video_frames: torch.FloatTensor
) -> torch.FloatTensor:
    video_frames = video_frames.to(device=vae.device, dtype=vae.dtype)
    video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    latent_dist = vae.encode(x=video_frames).latent_dist.sample().transpose(1, 2)
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
    image: torch.FloatTensor,
    video: torch.FloatTensor,
    scheduler,
    prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 6,
    preservation_scale: float = 3,
    use_dynamic_cfg: bool = False,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    strength: float = 0.8,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> torch.FloatTensor:
    pipeline._guidance_scale = guidance_scale
    pipeline._attention_kwargs = attention_kwargs
    pipeline._interrupt = False

    device = pipeline._execution_device

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # 2. Default call parameters
    num_videos_per_prompt = 1
    num_frames = 49

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

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
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    
    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)
    # added the following 2 lines
    timesteps, num_inference_steps = pipeline.get_timesteps(num_inference_steps, timesteps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)
    pipeline._num_timesteps = len(timesteps)

    # 5. Prepare latents.
    image = pipeline.video_processor.preprocess(image, height=height, width=width).to(
        device, dtype=prompt_embeds.dtype
    )

    # TODO: Added the following line ---
    # Process the video
    video = pipeline.video_processor.preprocess_video(video, height=height, width=width)
    video = video.to(device=device, dtype=prompt_embeds.dtype)

    latent_channels = pipeline.transformer.config.in_channels // 2
    # ------------------------------------------------------------ #
    latents, image_latents = pipeline.prepare_latents(
        image,
        video,
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
        latent_timestep,
        attention_kwargs
    )
    # ------------------------------------------------------------ #

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    if isinstance(
        scheduler, DDIMInverseScheduler
    ):  # Inverse scheduler does not accept extra kwargs
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
            
            attention_kwargs['timestep'] = i
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
           
            latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

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

            # perform guidance
            dynamic_scale = guidance_scale
            if use_dynamic_cfg:
                dynamic_scale = guidance_scale + 2 -  (1.0 + guidance_scale * ( # guidance_scale + 2 - (
                        (
                            1
                        - math.cos(
                            math.pi
                            * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0
                        )
                    )
                    / 2
                ))
                
            # print('dynamic_scale: ', dynamic_scale)
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # position-wise guidance scale
                noise_pred = torch.where(attention_kwargs['positions_to_replace'], 
                                        noise_pred_uncond + dynamic_scale * (noise_pred_text - noise_pred_uncond),
                                        noise_pred_uncond + preservation_scale * (noise_pred_text - noise_pred_uncond))

            # compute the noisy sample x_t-1 -> x_t
            # FIXIT: Problem is here
            latents = scheduler.step(
                model_output=noise_pred, timestep=t, sample=latents, **extra_step_kwargs, return_dict=False
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

# sample noise: torch.Size([1, 13, 16, 60, 90])
@torch.no_grad()
def inverse_dvs(
    model_path: str,
    prompt: str,
    video_path: str,
    output_path: str,
    guidance_scale: float,
    preservation_scale: float,
    num_inference_steps: int,
    width: int,
    height: int,
    fps: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
    lora_path: str = None,
    inverted_latent_path: str = None,
    mask_path: str = None,
    depth_path: str = None,
    k_order: int = 3,
    treshold_idx: int = -5,
):
    # set seed
    set_seed(seed)
    # Load the model
    pipeline: CogVideoXImageToVideoPipeline = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device=device)

    # Load the new transformer model
    transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, 
                                                              subfolder="transformer", 
                                                              torch_dtype=dtype).to(device=device)
    # Set the transformer to the new model
    pipeline.transformer = transformer

    # Set the scheduler to the new model
    pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config)
    print("--------------------------------")
    pipeline.scheduler.config['prediction_type'] = 'sample'
    print('prediction type: ', pipeline.scheduler.config['prediction_type'])
    print("--------------------------------")
    
    if not pipeline.transformer.config.use_rotary_positional_embeddings:
        raise NotImplementedError("This script supports CogVideoX 5B model only.")

    if lora_path != "None":
        print('Loading LoRA weights from: ', lora_path)
        base_url = 'Eyeline-Research/Go-with-the-Flow'
        lora_path = hf_hub_download(repo_id=base_url, filename=lora_path)
        pipeline.load_lora_weights(lora_path)

    # Get video and first frame image
    video = load_video(os.path.join(output_path, 'render.mp4'))
    image = video[0]
    
    inverted_latent = None
    if inverted_latent_path is not None:
        inverted_latent = torch.load(inverted_latent_path).to(device=device, dtype=dtype)
    
    depth = None
    if depth_path is not None:
        # load depth
        depth = np.load(depth_path)['depths']

        # normalize depth to 0-1
        max_values = depth.reshape(depth.shape[0], -1).max(axis=1)
        depth = depth / max_values[:, None, None] # normalized to 0-1

        # resize depth to 60 x 90
        scale_h = 60 / depth.shape[1]
        scale_w = 90 / depth.shape[2]
        interpolated_depth = zoom(depth, (1, scale_h, scale_w), order=1)  # order=1 for bilinear interpolation

        # binary interpolation
        binary_interpolated_depth = np.where(interpolated_depth < 0.3, 1, 0)
        binary_interpolated_depth_downsized = rp.resize_list(binary_interpolated_depth, 13) 
        depth = torch.from_numpy(binary_interpolated_depth_downsized).to(device=device, dtype=dtype).unsqueeze(0) # 1 x 13 x 60 x 90, normalized to 0-1 and 1s region are the backgrounds
  

    # Load mask 
    if mask_path is not None:
        mask = torch.load(mask_path).to(device=device, dtype=dtype)
        mask_resized = F.interpolate(mask, size=(480, 720), mode='area') # 49 x 1 x 480 x 720
        print('[done processing mask...]')
        

        mask_resized = mask_resized.unsqueeze(0).transpose(1, 2)
        mask_resized = downsize_mask(mask_resized).transpose(1, 2).repeat(1, 1, 16, 1, 1)
        treshold = mask_resized.unique()[-treshold_idx]
        print('mask_resized: ', mask_resized.unique())
        
        mask_resized_branch = mask_resized.clone()
        mask_resized_branch = torch.where(mask_resized_branch <= treshold, 0., 1.)  
        correct_motion = (mask_resized_branch == 0.)

        mask_resized = torch.where(mask_resized < 1.0, 0., 1.)  
        positions_to_replace = (mask_resized == 0.)
        print('positions_to_replace: ', positions_to_replace.shape)
        
        
    attention_kwargs = {'layer': 0, 
                        'timestep': None, 
                        'positions_to_replace': positions_to_replace, 
                        'correct_motion': correct_motion, 
                        'depth': depth, 
                        'k_order': k_order} 

    with torch.no_grad():
        recon_latents = sample(
                    pipeline=pipeline,
                    latents=inverted_latent[0],
                    image=image,
                    video=video,
                    strength=0.98,
                    scheduler=pipeline.scheduler,
                    prompt=prompt,
                    negative_prompt='crowded scene, car in the road, bad anatomy, deformed body, occlusions, dark scene, black, unseen regions, black & white, blurry, pixelated.', #"low quality, low resolution, blurry, pixelated, jpeg artifacts, compression artifacts, bad anatomy, deformed body, disproportionate body, distorted limbs, bad proportions, extra limbs, missing limbs, floating limbs, disconnected limbs, mutation, mutated hands and fingers, extra fingers, fused fingers, too many fingers, missing fingers, bad hands, poorly drawn hands, malformed hands, broken hands, duplicate body parts, amputated limbs, disfigured, malformed, mutated, anatomical nonsense, bad composition, cropped image, frame cut, out of frame, poorly framed, over saturation, under saturation, over exposed, under exposed, washed out colors, dull colors, grainy, noisy, watermark, text, signature, logo, username, bad lighting, harsh shadows, unnatural shadows, poor lighting, unnatural lighting, amateur, unprofessional, poorly drawn face, deformed face, ugly, cross-eyed, squinting, grimacing, distorted face, unnatural face, asymmetric face",
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    preservation_scale=preservation_scale,
                    generator=torch.Generator(device=device).manual_seed(seed),
                    height=height,
                    width=width,
                    use_dynamic_cfg=True, 
                    attention_kwargs=attention_kwargs,
                    )
    # Create output video
    extension = f't:{num_inference_steps}_k:{k_order}_g:{guidance_scale}_p:{preservation_scale}_t:{treshold_idx}.mp4'
    output_video_name = f'output_{extension}'
    recon_video_path = os.path.join(output_path, output_video_name)
    export_latents_to_video(pipeline, recon_latents[-1], recon_video_path, fps)
    # Create grid video
    input_path = os.path.join(output_path, "input.mp4")
    comparison_video_name = f"comparison_1x2_{extension}"
    comparison_output_path = os.path.join(output_path, comparison_video_name)
    create_video_grid(video_paths=[input_path, recon_video_path], titles=["Input", "Output"], output_path=comparison_output_path, grid_size="1x2")


if __name__ == "__main__":
    arguments = get_args()
    inverse_dvs(**arguments)