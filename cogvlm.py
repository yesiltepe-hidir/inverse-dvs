import io
import os
import argparse
import numpy as np
import torch
from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "THUDM/cogvlm2-llama3-caption"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")
parser.add_argument('--quant', type=int, choices=[4, 8], help='Enable 4-bit or 8-bit precision loading', default=0)
parser.add_argument('--video_path', type=str, help='Path to the video file')
parser.add_argument('--prompts_dir', type=str, default='prompts', help='Directory to save the generated prompt')
parser.add_argument('--custom_prompt', type=str, default='Please describe this video in detail.', help='Custom prompt to use')
args = parser.parse_args()


def load_video(video_data, strategy='chat'):
    bridge.set_bridge('torch')
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == 'base':
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = min(total_frames,
                        int(clip_end_sec * decord_vr.get_avg_fps())) if clip_end_sec is not None else total_frames
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == 'chat':
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True
).eval().to(DEVICE)


def predict(prompt, video_data, temperature):
    strategy = 'chat'

    video = load_video(video_data, strategy=strategy)

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version=strategy
    )
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(TORCH_TYPE)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": False,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def process_video(video_path, prompts_dir, custom_prompt=None):
    # Ensure prompts directory exists
    os.makedirs(prompts_dir, exist_ok=True)
    
    # Extract video name from path
    video_name = video_path.split("/")[-2]
    # Path to save the prompt
    prompt_file = os.path.join(prompts_dir, f"{video_name}.txt")

    print(f"Processing video: {video_path}")
    print(f"Will save prompt to: {prompt_file}")
    
    # Generate prompt
    # prompt = "Give the comprehensive prompt that generates this scene. Give explanatory details about the possible place that this scene is located. Discard the black regions due to camera motion while describing the scene. In the video, you will see black, unseen regions due to camera motion. DO NOT MENTION/DESCRIBE THESE BLACK UNSEEN REGIONS.  Instead, HALLUCINATE what would be in these black regions in great detail. Give an inpainting prompt for the black regions by describing the surroundings of the scene. Hallucinate the surroundings in great detail. Add some environmental details to the hallucination. DO NOT TYPE THE WORD 'BLACK' IN THE PROMPT. DO NOT MENTION THE WORD 'BLACK' IN THE PROMPT. DO NOT SPECIFY THE CAMERA TYPE (e.g. 'fisher camera', 'handheld camera', 'drone', 'satellite', etc.) IN THE PROMPT. Describe what is becoming visible as the camera moves. State explicitly that the 'as the camera moves' and then describe what is becoming visible."
    prompt = "describe the scene in detail. Describe the actions and surroundings."
    temperature = 0.1
    
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        # Get description from model
        response = predict(prompt, video_data, temperature)

        # Save to file
        print("Saving to file:", prompt_file)
        with open(prompt_file, 'w') as f:
            f.write(response)
        
        print(f"Successfully generated and saved prompt for {video_name}")
        return True
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return False


if __name__ == '__main__':
    if args.video_path:
        process_video(args.video_path, args.prompts_dir, args.custom_prompt)
    else:
        print("No video path provided. Please use --video_path to specify a video file.")

# transformers library will be changed from '4.50.1' to '4.48.3'