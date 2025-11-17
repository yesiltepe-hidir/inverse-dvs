
import numpy as np
import cv2
import PIL
from PIL import Image
import os
from datetime import datetime
import pdb
import torch.nn.functional as F
import numpy as np
import os
import copy
from scipy.interpolate import UnivariateSpline, interp1d
import numpy as np
import PIL.Image
import torchvision
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
import cv2
import PIL
import numpy
import skimage.io
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
import argparse
from diffusers.training_utils import set_seed
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
import rp
def save_video(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [
            np.array(Image.open(os.path.join(folder_name, path)))
            for folder_name, path in zip(folder, data)
        ]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(
        images_path, tensor_data, fps=fps, video_codec='h264', options={'crf': '10'}
    )

def sphere2pose(c2ws_input, theta, phi, r, device, x=None, y=None):
    c2ws = copy.deepcopy(c2ws_input)
    # c2ws[:,2, 3] = c2ws[:,2, 3] - radius

    # 先沿着世界坐标系z轴方向平移再旋转
    c2ws[:, 2, 3] -= r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] -= x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, cos_value_x, -sin_value_x, 0],
                [0, sin_value_x, cos_value_x, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = (
        torch.tensor(
            [
                [cos_value_y, 0, sin_value_y, 0],
                [0, 1, 0, 0],
                [-sin_value_y, 0, cos_value_y, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)
    # c2ws[:,2, 3] = c2ws[:,2, 3] + radius
    return c2ws

def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew

def generate_traj_txt(c2ws_anchor, phi, theta, r, frame, device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    if len(phi) > 3:
        phis = txt_interpolation(phi, frame, mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi, frame, mode='linear')

    if len(theta) > 3:
        thetas = txt_interpolation(theta, frame, mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta, frame, mode='linear')

    if len(r) > 3:
        rs = txt_interpolation(r, frame, mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]
    else:
        rs = txt_interpolation(r, frame, mode='linear')
    # rs = rs*c2ws_anchor[0,2,3].cpu().numpy()

    c2ws_list = []
    for th, ph, r in zip(thetas, phis, rs):
        c2w_new = sphere2pose(
            c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws

def read_video_frames(video_path, process_length, stride, dataset="open", opts=None):
    if dataset == "open":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
        # original_height, original_width = vid.get_batch([0]).shape[1:3]
        # height = round(original_height / 64) * 64
        # width = round(original_width / 64) * 64
        # if max(height, width) > max_res:
        #     scale = max_res / max(original_height, original_width)
        #     height = round(original_height * scale / 64) * 64
        #     width = round(original_width * scale / 64) * 64

        # FIXME: hard coded
        width = 1024
        height = 576

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    # Update the video length
    opts.video_length = len(frames_idx)
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    return frames

# ------------------------------------------ DepthCrafter ------------------------------------------------------------ #
class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
        device: str = "cuda:0",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to(device)
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        frames,
        near,
        far,
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
        depth_path: str = None,
    ):
        set_seed(seed)

        # inference the depth map using the DepthCrafter pipeline
        # with torch.inference_mode():
        #     res = self.pipe(
        #         frames,
        #         height=frames.shape[1],
        #         width=frames.shape[2],
        #         output_type="np",
        #         guidance_scale=guidance_scale,
        #         num_inference_steps=num_denoising_steps,
        #         window_size=window_size,
        #         overlap=overlap,
        #         track_time=track_time,
        #     ).frames[0]
        # # convert the three-channel output to a single channel depth map
        # res = res.sum(-1) / res.shape[-1]
        assert depth_path is not None, "depth_path is required"
        res = np.load(depth_path)["depths"]

        # normalize the depth map to [0, 1] across the whole video
        depths = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        # vis = vis_sequence_depth(res)
        # save the depth map and visualization with the target FPS
        depths = torch.from_numpy(depths).unsqueeze(1)  # 49 576 1024 ->
        depths *= 3900  # compatible with da output
        depths[depths < 1e-5] = 1e-5
        depths = 10000.0 / depths
        depths = depths.clip(near, far)

        return depths
# ------------------------------------------ Warper ------------------------------------------------------------ #
class Warper:
    def __init__(self, resolution: tuple = None, device: str = 'gpu0'):
        self.resolution = resolution
        self.device = self.get_device(device)
        self.dtype = torch.float32
        return

    def forward_warp(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
        mask=False,
        twice=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device).to(self.dtype)
        mask1 = mask1.to(self.device).to(self.dtype)
        depth1 = depth1.to(self.device).to(self.dtype)
        transformation1 = transformation1.to(self.device).to(self.dtype)
        transformation2 = transformation2.to(self.device).to(self.dtype)
        intrinsic1 = intrinsic1.to(self.device).to(self.dtype)
        intrinsic2 = intrinsic2.to(self.device).to(self.dtype)

        trans_points1 = self.compute_transformed_points(
            depth1, transformation1, transformation2, intrinsic1, intrinsic2
        )
        trans_coordinates = (
            trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        )
        trans_depth1 = trans_points1[:, :, :, 2, 0]
        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid
        if not twice:
            warped_frame2, mask2 = self.bilinear_splatting(
                frame1, mask1, trans_depth1, flow12, None, is_image=True
            )
            if mask:
                warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)
            return warped_frame2, mask2, None, flow12

        else:
            warped_frame2, mask2 = self.bilinear_splatting(
                frame1, mask1, trans_depth1, flow12, None, is_image=True
            )
            # warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)
            warped_flow, _ = self.bilinear_splatting(
                flow12, mask1, trans_depth1, flow12, None, is_image=False
            )
            twice_warped_frame1, _ = self.bilinear_splatting(
                warped_frame2,
                mask2,
                depth1.squeeze(1),
                -warped_flow,
                None,
                is_image=True,
            )
            return twice_warped_frame1, warped_frame2, None, None

    def compute_transformed_points(
        self,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
    ):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        transformation = torch.bmm(
            transformation2, torch.linalg.inv(transformation1)
        )  # (b, 4, 4)

        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat(
            [b, 1, 1, 1, 1]
        )  # (b, h, w, 1, 1)
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[
            None, :, :, :, None
        ]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(
            intrinsic1_inv_4d, pos_vectors_homo
        )  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    def bilinear_splatting(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor],
        is_image: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack(
            [
                torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_floor = torch.stack(
            [
                torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        trans_pos_ceil = torch.stack(
            [
                torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(
            prox_weight_nw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_sw = torch.moveaxis(
            prox_weight_sw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_ne = torch.moveaxis(
            prox_weight_ne * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        weight_se = torch.moveaxis(
            prox_weight_se * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(
            frame1
        )
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(
            frame1
        )

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_nw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_sw,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_ne,
            accumulate=True,
        )
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_se,
            accumulate=True,
        )

        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            weight_nw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            weight_sw,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            weight_ne,
            accumulate=True,
        )
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            weight_se,
            accumulate=True,
        )

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(
            mask, cropped_warped_frame / cropped_weights, zero_tensor
        )
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, mask2

    def clean_points(self, warped_frame2, mask2):
        warped_frame2 = (warped_frame2 + 1.0) / 2.0
        mask = 1 - mask2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0) * 255.0
        mask = mask.cpu().numpy()
        kernel = numpy.ones((5, 5), numpy.uint8)
        mask_erosion = cv2.dilate(numpy.array(mask), kernel, iterations=1)
        mask_erosion = PIL.Image.fromarray(numpy.uint8(mask_erosion))
        mask_erosion_ = numpy.array(mask_erosion) / 255.0
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1
        mask_new = (
            torch.from_numpy(mask_erosion_)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        warped_frame2 = warped_frame2 * (1 - mask_new)
        return warped_frame2 * 2.0 - 1.0, 1 - mask_new[:, 0:1, :, :]

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def read_image(path: Path) -> torch.Tensor:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def read_depth(path: Path) -> torch.Tensor:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        else:
            raise RuntimeError(f'Unknown depth format: {path.suffix}')
        return depth

    @staticmethod
    def camera_intrinsic_transform(
        capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)
    ):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(4)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        if device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device('cpu')
        return device

# ------------------------------------------ TrajectoryWarper ------------------------------------------------------------ #
class TrajectoryWarper:
    def __init__(self, opts):
        self.depth_estimater = DepthCrafterDemo(
            unet_path=opts.unet_path,
            pre_train_path=opts.pre_train_path,
            cpu_offload=opts.cpu_offload,
            device=opts.device,
        )
        self.funwarp = Warper(device=opts.device)
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(opts.blip_path, torch_dtype=torch.float32).to(opts.device)
        self.opts = opts
    
    def get_caption(self, opts, image):
        image_array = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
            opts.device, torch.float32
        )
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return generated_text + opts.refine_prompt

    def get_poses(self, opts, depths, num_frames):
        radius = (
            depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu()
            * opts.radius_scale
        )
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (
            torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            .repeat(num_frames, 1, 1)
            .to(opts.device)
        )
        c2w_init = (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            .to(opts.device)
            .unsqueeze(0)
        )
        # TODO: Here
        if self.opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(
                c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device
            )
        elif self.opts.camera == 'traj':
            with open(self.opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[self.opts.anchor_idx : self.opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K

    def infer(self):
        # Get the frames
        frames = read_video_frames(
            self.opts.video_path, self.opts.video_length, self.opts.stride, opts=self.opts
        )
        # Get the caption of the middle frame
        # prompt = self.get_caption(self.opts, frames[self.opts.video_length // 2])

        # Estimate the depths of the frames
        depths = self.depth_estimater.infer(
                    frames,
                    self.opts.near,
                    self.opts.far,
                    self.opts.depth_inference_steps,
                    self.opts.depth_guidance_scale,
                    window_size=self.opts.window_size,
                    overlap=self.opts.overlap,
                    depth_path=self.opts.depth_path,
                ).to(self.opts.device)
        
        # Convert the frames to the correct format
        frames = (
            torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.opts.device) * 2.0 - 1.0
        )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == self.opts.video_length

        # Get the poses
        pose_s, pose_t, K = self.get_poses(self.opts, depths, num_frames=self.opts.video_length)
        
        # Warp the frames
        warped_images = []
        masks = []

        # Warp the frames
        for i in tqdm(range(self.opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i : i + 1],
                None,
                depths[i : i + 1],
                pose_s[i : i + 1],
                pose_t[i : i + 1],
                K[i : i + 1],
                None,
                self.opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
    
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks_original = torch.cat(masks)
        # Resize using nearest neighbor interpolation to preserve boolean values
        # cond_masks = F.interpolate(cond_masks_original, size=(30, 45), mode='nearest')
        torch.save(cond_masks_original, os.path.join(self.opts.save_dir, 'mask.pt'))

        frames = F.interpolate(
            frames, size=self.opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_video = F.interpolate(
            cond_video, size=self.opts.sample_size, mode='bilinear', align_corners=False
        )
        cond_masks_original = F.interpolate(cond_masks_original, size=self.opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(self.opts.save_dir, 'control_input.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'render.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_masks_original.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'mask.mp4'),
            fps=self.opts.fps,
        )

        # Save the prompt
        # with open(os.path.join(self.opts.save_dir, 'prompt.txt'), 'w') as f:
        #     f.write(prompt)

# ------------------------------------------ Parser ------------------------------------------------------------ #
def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Input/output arguments
    parser.add_argument(
        '--video_path',
        type=str,
        default='./davis_data/videos/blackswan.mp4',
        help='Path to input video file'
    )
    parser.add_argument(
        '--save_dir', 
        type=str,
        default='./result',
        help='Directory to save outputs'
    )

    # Video processing arguments
    parser.add_argument(
        '--video_length',
        type=int,
        default=46,
        help='Number of frames to process\nDefault: 49'
    )
    parser.add_argument(
        '--stride',
        type=int, 
        default=1,
        help='Stride between processed frames\nDefault: 1'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Output video framerate\nDefault: 12'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        nargs=2,
        default=[480, 720],
        help='Output resolution as [height, width]\nDefault: [384, 672]'
    )

    # Model arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run model on (cuda/cpu)\nDefault: cuda'
    )
    parser.add_argument(
        '--unet_path',
        type=str, 
        default='tencent/DepthCrafter',
        help='Path to UNet model\nDefault: tencent/DepthCrafter'
    )
    parser.add_argument(
        '--pre_train_path',
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help='Path to pre-trained model\nDefault: stabilityai/stable-video-diffusion-img2vid-xt'
    )
    parser.add_argument(
        '--cpu_offload',
        type=str,
        default='model',
        help='CPU offload strategy\nDefault: model'
    )

    # Depth estimation arguments
    parser.add_argument(
        '--near',
        type=float,
        default=0.0001, # 0.0001
        help='Near clipping plane distance\nDefault: 0.0001'
    )
    parser.add_argument(
        '--far',
        type=float,
        default=10000.0, # 10000.0
        help='Far clipping plane distance\nDefault: 10000.0'
    )
    parser.add_argument(
        '--depth_inference_steps',
        type=int,
        default=5,
        help='Number of inference steps\nDefault: 5'
    )
    parser.add_argument(
        '--depth_guidance_scale',
        type=float,
        default=1.0,
        help='Guidance scale for depth estimation\nDefault: 1.0'
    )

    # Processing window arguments
    parser.add_argument(
        '--window_size',
        type=int,
        default=110,
        help='Window size for processing\nDefault: 110'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=49,
        help='Overlap between processing windows\nDefault: 25'
    )
    parser.add_argument(
        '--mask',
        action='store_true',
        default=False,
        help='Enable masking\nDefault: False'
    )

    parser.add_argument(
        '--camera',
        type=str,
        default='traj',
        help='Camera type'
    )

    parser.add_argument(
        '--anchor_idx',
        type=int,
        default=0,
        help='Anchor index'
    )

    parser.add_argument(
        '--radius_scale',
        type=float,
        default=1.0,
        help='Scale factor for the spherical radius',
    )

    parser.add_argument(
        '--target_pose',
        nargs=5,
        type=float,
        help="Required for 'target' mode, specify target camera pose, <theta phi r x y>",
    )

    parser.add_argument(
        '--traj_txt',
        type=str,
        default='./traj.txt',
        help='Path to trajectory text file',
    )
    
    parser.add_argument(
        '--refine_prompt',
        type=str,
        default='',
        help='Refine prompt',
    )

    parser.add_argument(
        '--blip_path',
        type=str,
        default='Salesforce/blip2-opt-2.7b',
        help='Path to BLIP model',
    )

    parser.add_argument(
        '--depth_path',
        type=str,
        default=None,
        help='Path to depth map',
    )
    return parser
# ------------------------------------------ Main ------------------------------------------------------------ #
if __name__ == "__main__":
    # Get the arguments
    parser = get_parser()
    opts = parser.parse_args()

    # Initialize the trajectory warper
    trajectory_warper = TrajectoryWarper(opts)

    # Infer the trajectory
    trajectory_warper.infer()
