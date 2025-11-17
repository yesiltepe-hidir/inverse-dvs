#!/bin/bash

conda activate netflix

model_path="THUDM/CogVideoX-5b-I2V"
lora_path="I2V5B_final_i38800_nearest_lora_weights.safetensors"

# Create inversions directory if it doesn't exist
mkdir -p inversions
output_path="inversions"

# Define subfolders to process
subfolders=("loop1")

# Create array of GPU IDs
gpu_ids=(0 1 3 5 6 7)
num_inference_steps=(20 30 50 100 250 500)
num_gpus=${#gpu_ids[@]}
counter=0
data_root_path="warped"

# # Convert videos to frames first
# for subfolder in "${subfolders[@]}"; do
#     # Define video files for this subfolder
#     video_files=($data_root_path/$subfolder/*/render.mp4)
    
#     for video_path in "${video_files[@]}"; do
#         if [ ! -f "$video_path" ]; then
#             continue
#         fi
        
#         video_name=$(basename "$video_path" .mp4)
#         video_dir=$(dirname "$video_path")
#         dir_name=$(basename "$video_dir")
        
#         # Check if frames directory exists
#         if [ ! -d "frames/render_${subfolder}_${dir_name}" ]; then
#             echo "Converting video to frames for $video_path"
#             python utils/video2frames.py --video_root_path "$data_root_path/$subfolder"
#         fi
#     done
# done

# Process each video
for subfolder in "${subfolders[@]}"; do
    # Define video files for this subfolder
    video_files=($data_root_path/$subfolder/*/render.mp4)
    
    for video_path in "${video_files[@]}"; do
        if [ ! -f "$video_path" ]; then
            continue
        fi
        
        video_name=$(basename "$video_path" .mp4)
        video_dir=$(dirname "$video_path")
        dir_name=$(basename "$video_dir")
        
        for num_inference_step in "${num_inference_steps[@]}"; do
            # Get GPU ID for this iteration
            gpu=${gpu_ids[$((counter % num_gpus))]}
            
            echo "Processing $video_path on GPU $gpu"
            
            # Check if output already exists
            if [ -f "$output_path/${subfolder}_latents_${dir_name}_${num_inference_step}.pt" ]; then
                echo "Skipping $video_path because it already exists"
                continue
            fi
            
            prompt=""
            echo "Prompt: $prompt"
            
            # Run on specific GPU in background
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            CUDA_VISIBLE_DEVICES=$gpu python i2v_ddim_inversion.py \
                --model_path "$model_path" \
                --prompt "$prompt" \
                --video_path "$video_path" \
                --output_path "$output_path" \
                --lora_path "$lora_path" \
                --subfolder "$subfolder" \
                --num_inference_steps $num_inference_step &
            
            # Increment counter
            ((counter++))
            
            # If we've started processes on all GPUs, wait for them to complete
            if [ $((counter % num_gpus)) -eq 0 ]; then
                wait
            fi
        done
    done
done

# Wait for all background processes to complete
wait
