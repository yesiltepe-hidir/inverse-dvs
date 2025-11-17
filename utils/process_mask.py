import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

def process_mask(mask):
    # Get the dimensions of the mask
    num_frames, channels, height, width = mask.shape

    # Process each frame
    for frame_idx in tqdm(range(num_frames)):
        # Get the current frame mask
        current_mask = mask[frame_idx, 0]  # Get the first channel
        
        # Create a mask to track pixels that need to be set to 0
        pixels_to_zero = torch.zeros_like(current_mask, dtype=torch.bool)
        
        # Define kernel size
        kernel_size = 5
        
        # Check each pixel's neighborhood with a 5x5 kernel
        for h in range(height):
            for w in range(width):
                # Only process pixels that are currently 0
                if current_mask[h, w] == 0:
                    # Define kernel boundaries
                    h_start = max(0, h - kernel_size // 2)
                    h_end = min(height, h + kernel_size // 2 + kernel_size % 2)
                    w_start = max(0, w - kernel_size // 2)
                    w_end = min(width, w + kernel_size // 2 + kernel_size % 2)
                    
                    # Set all pixels in the kernel region to 0
                    pixels_to_zero[h_start:h_end, w_start:w_end] = True
        
        # Apply the changes to the processed mask
        mask[frame_idx, 0][pixels_to_zero] = 0

    return mask


def display_mask(mask):
    mask_image = Image.fromarray((mask.squeeze().numpy() * 255).astype(np.uint8))
    display(mask_image)