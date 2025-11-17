import cv2
import numpy as np
import os
from typing import List
from diffusers.utils import export_to_video
from PIL import Image, ImageDraw, ImageFont
import argparse
def create_video_grid(video_paths: List[str], titles: List[str], output_path: str, grid_size: str) -> None:
    """
    Create a grid of videos side by side
    Args:
        video_paths: List of paths to video files
        titles: List of titles for each video
        output_path: Path to save the output video
        grid_size: String specifying grid layout ("1x2" or "2x2")
    """
    # Target dimensions
    TARGET_HEIGHT = 480
    TARGET_WIDTH = 730
    TITLE_HEIGHT = 50  # Increased height for larger font
    FONT_SIZE = 36  # Large font size
    
    # Parse grid size
    grid_size = grid_size.replace(" ", "").lower()  # Remove spaces and convert to lowercase
    if grid_size not in ["1x2", "2x2"]:
        raise ValueError("Grid size must be either '1x2' or '2x2'")
    
    rows, cols = map(int, grid_size.split("x"))
    
    # Check if all video files exist
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
    
    # Open all video captures
    captures = []
    for path in video_paths:
        cap = cv2.VideoCapture(str(path))  # Convert path to string
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")
        captures.append(cap)
    
    # Store all frames for export
    all_frames = []
    
    # Process each frame
    while True:
        frames = []
        for cap in captures:
            ret, frame = cap.read()
            if not ret:
                # If any video ends, we're done
                for c in captures:
                    c.release()
                # Convert frames to numpy array and export
                video_frames = np.stack(all_frames[:-1] + all_frames[::-1])
                export_to_video(video_frames, output_path, fps=30)
                return
                
            # Convert BGR to RGB and normalize to 0-1 range
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            
            # Resize frame
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            frames.append(frame)
        

        if grid_size == "1x2":
            # Combine frames horizontally for 1x2
            combined_frame = np.hstack(frames)
            # Create a black space for titles
            title_space = np.zeros((TITLE_HEIGHT, combined_frame.shape[1], 3), dtype=np.float32)
            # Add title space to the frame
            full_frame = np.vstack([title_space, combined_frame])
        else:  # 2x2
            # Create title spaces for both rows
            title_space = np.zeros((TITLE_HEIGHT, TARGET_WIDTH * 2, 3), dtype=np.float32)
            
            # Split frames into two rows and add title space to each
            row1 = np.hstack(frames[:2])
            row1_with_title = np.vstack([title_space, row1])
            
            row2 = np.hstack(frames[2:])
            row2_with_title = np.vstack([title_space, row2])
            
            # Combine both rows
            full_frame = np.vstack([row1_with_title, row2_with_title])
        
        # Convert to PIL Image for text overlay
        pil_image = Image.fromarray((full_frame * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a larger font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", FONT_SIZE)
            except:
                font = ImageFont.load_default()
        
        # Add centered titles using PIL
        if grid_size == "1x2":
            for i, title in enumerate(titles):
                text_width = draw.textlength(title, font=font)
                x = i * TARGET_WIDTH + (TARGET_WIDTH - text_width) // 2
                y = (TITLE_HEIGHT - FONT_SIZE) // 2
                draw.text((x, y), title, fill=(255, 255, 255), font=font)
        else:  # 2x2
            for i, title in enumerate(titles):
                row = i // 2
                col = i % 2
                text_width = draw.textlength(title, font=font)
                x = col * TARGET_WIDTH + (TARGET_WIDTH - text_width) // 2
                y = row * (TARGET_HEIGHT + TITLE_HEIGHT) + (TITLE_HEIGHT - FONT_SIZE) // 2
                draw.text((x, y), title, fill=(255, 255, 255), font=font)
        
        # Convert back to numpy array in 0-1 range
        combined_frame = np.array(pil_image).astype(np.float32) / 255.0
        all_frames.append(combined_frame)
    
    # Clean up
    for cap in captures:
        cap.release()

if __name__ == "__main__":
    import sys
    
    
        
    parser = argparse.ArgumentParser(description='Create a video grid from multiple videos')
    parser.add_argument('--video_dir', help='Directory containing the video files')
    parser.add_argument('--grid_size', choices=['1x2', '2x2'], help='Grid size (1x2 or 2x2)')
    parser.add_argument('--guidance_scale', type=float, help='Guidance scale value')
    parser.add_argument('--num_inference_steps', type=int, help='Number of inference steps')
    parser.add_argument('--block_indexes', type=str, help='Block indexes')
    
    args = parser.parse_args()
    video_dir = args.video_dir
    grid_size = args.grid_size
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    block_indexes = args.block_indexes
    input_path = os.path.join(video_dir, "input.mp4")
    output_path = os.path.join(video_dir, "output_t:"+str(num_inference_steps)+"_g:"+str(guidance_scale)+"_b:"+str(block_indexes)+".mp4")
    conditional_input_path = os.path.join(video_dir, "control_input_t:"+str(num_inference_steps)+"_g:"+str(guidance_scale)+".mp4")
    render_path = os.path.join(video_dir, "render_t:"+str(num_inference_steps)+"_g:"+str(guidance_scale)+".mp4")
    
    # Check if files exist based on grid size
    if grid_size.replace(" ", "").lower() == "1x2":
        if not os.path.exists(input_path) or not os.path.exists(output_path):
            print(f"Error: Both input.mp4 and output.mp4 must exist in {video_dir}")
            sys.exit(1)
        video_paths = [input_path, output_path]
        titles = ["Input", "Output"]
    else:  # 2x2
        if not all(os.path.exists(p) for p in [input_path, conditional_input_path, output_path, render_path]):
            print(f"Error: input.mp4, control_input.mp4, and output.mp4 must exist in {video_dir}")
            sys.exit(1)
        video_paths = [input_path, output_path, conditional_input_path, render_path]
        titles = ["Input", "Output", "Control", "Render"]
    
    output_name = f"comparison_1x2_g:{guidance_scale}_t:{num_inference_steps}.mp4" if grid_size == "1x2" else f"comparison_2x2_g:{guidance_scale}_t:{num_inference_steps}.mp4"
    save_path = os.path.join(video_dir, output_name)
    create_video_grid(video_paths, titles, save_path, grid_size)
