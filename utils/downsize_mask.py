
import torch
import torch.nn.functional as F


def downsize_mask(x: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, num_frames, height, width = x.shape


    frame_batch_size = 8
    # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
    # As the extra single frame is handled inside the loop, it is not required to round up here.
    num_batches = max(num_frames // frame_batch_size, 1)
    enc = []

    for i in range(num_batches):
        remaining_frames = num_frames % frame_batch_size
        start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
        end_frame = frame_batch_size * (i + 1) + remaining_frames
        x_intermediate = x[:, :, start_frame:end_frame]
        # 1: Time Compression
        x_intermediate = encoder(x_intermediate, compress_time=True)
        # 2: 2 Time Compression
        x_intermediate = encoder(x_intermediate, compress_time=True)
        # 3: Spatial Compression
        x_intermediate = encoder(x_intermediate, compress_time=False)
        enc.append(x_intermediate)

    enc = torch.cat(enc, dim=2)
    return enc

# Then 3 times following code
# 1: Time Compression
# 2: 2 Time Compression
# 3: Spatial Compression
def encoder(x: torch.Tensor, compress_time: bool = True) -> torch.Tensor:
    if compress_time:
        batch_size, channels, frames, height, width = x.shape

        # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
        x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

        if x.shape[-1] % 2 == 1:
            x_first, x_rest = x[..., 0], x[..., 1:]
            if x_rest.shape[-1] > 0:
                # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

            x = torch.cat([x_first[..., None], x_rest], dim=-1)
            # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
            x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
        else:
            # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
            # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
            x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

    # Pad the tensor
    pad = (0, 1, 0, 1)
    x = F.pad(x, pad, mode="constant", value=0)
    batch_size, channels, frames, height, width = x.shape
    # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
    x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
    x = F.avg_pool2d(x, kernel_size=3, stride=2) #self.conv(x) # kernel_size = 3, stride = 2
    # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
    x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
    return x


if __name__ == "__main__":
    mask_path = '/home/grads/hidir/camera-control/warped/loop1/11_monkey/mask.pt' # [49, 1, 576, 1024]
    mask = torch.load(mask_path) # [49, 1, 576, 1024] 
    mask_resized = F.interpolate(mask, size=(480, 720), mode='area') # [49, 1, 480, 720]
    # batch_size, num_channels, num_frames, height, width
    mask_resized = mask_resized.unsqueeze(0).transpose(1, 2) # [1, 49, 1, 480, 720]
    print(mask_resized.shape)
    mask_downsized = downsize_mask(mask_resized).transpose(1, 2).repeat(1, 1, 16, 1, 1) # [1, 13, 1, 60, 90]
    print(mask_downsized.shape)