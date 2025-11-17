#!/bin/bash

set -e

# Parse arguments
# Check if subfolder is provided as the first argument
if [ "$1" != "" ]; then
    SINGLE_SUBFOLDER="$1"
    echo "Processing single subfolder: $SINGLE_SUBFOLDER"
fi

# Check if config file is provided as the second argument
CONFIG_FILE="config.yml"
if [ "$2" != "" ]; then
    CONFIG_FILE="$2"
    echo "Using config file: $CONFIG_FILE"
fi

# Check if config.yml exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found. Please make sure it exists in the current directory."
    exit 1
fi

# Check if parse_yaml.py exists and is executable
if [ ! -x "parse_yaml.py" ]; then
    echo "Error: parse_yaml.py not found or not executable. Please make sure it exists and has execute permissions."
    exit 1
fi

CONFIG_GPU_IDS=($(./parse_yaml.py "$CONFIG_FILE" gpu.device_ids[]))

# Check if special value [-1] is provided (use all available GPUs)
if [ "${#CONFIG_GPU_IDS[@]}" -eq 1 ] && [ "${CONFIG_GPU_IDS[0]}" -eq -1 ]; then
    echo "Config specifies [-1] for GPU IDs: finding available GPUs..."
    # Get available GPUs using our utility
    GPU_DEVICE_IDS=($(python3 utils/check_gpus.py -1))
    if [ "${#GPU_DEVICE_IDS[@]}" -eq 0 ]; then
        echo "No available GPUs found. Using GPU 0 as fallback."
        GPU_DEVICE_IDS=(0)
    else
        echo "Found ${#GPU_DEVICE_IDS[@]} available GPUs: ${GPU_DEVICE_IDS[@]}"
    fi
else
    # Use the configured GPU IDs
    GPU_DEVICE_IDS=("${CONFIG_GPU_IDS[@]}")
fi

# Use experiment_dir instead of dataset.root_dir
EXPERIMENT_DIR=$(./parse_yaml.py "$CONFIG_FILE" common.experiment_dir)
ORIGINAL_ROOT_DIR=$(./parse_yaml.py "$CONFIG_FILE" dataset.root_dir)
VIDEO_PATTERN=$(./parse_yaml.py "$CONFIG_FILE" dataset.video_pattern)
SPECIFIC_VIDEOS=($(./parse_yaml.py "$CONFIG_FILE" dataset.specific_videos[]))
MODEL_PATH=$(./parse_yaml.py "$CONFIG_FILE" inference.model_path)
LORA_PATH=$(./parse_yaml.py "$CONFIG_FILE" inference.lora_path)
INVERSION_DIR=$(./parse_yaml.py "$CONFIG_FILE" inference.inversion_dir)
NUM_INFERENCE_STEPS=($(./parse_yaml.py "$CONFIG_FILE" inference.num_inference_steps[]))
PROMPTS_DIR=$(./parse_yaml.py "$CONFIG_FILE" common.prompts_dir)
SEED=$(./parse_yaml.py "$CONFIG_FILE" inference.seed)

# Get subfolders to process
if [ -n "$SINGLE_SUBFOLDER" ]; then
    # Use the provided subfolder
    SUBFOLDERS=("$SINGLE_SUBFOLDER")
else
    # Get all subfolders from config
    SUBFOLDERS=($(./parse_yaml.py "$CONFIG_FILE" experiment.subfolders[]))
fi

# Create inversions directory if it doesn't exist
mkdir -p "$INVERSION_DIR"

# Get the number of available GPUs for parallel processing
NUM_GPUS=${#GPU_DEVICE_IDS[@]}
echo "Will use $NUM_GPUS GPUs in parallel: ${GPU_DEVICE_IDS[*]}"

# Function to handle interruptions and clean up
function cleanup {
    echo "Cleaning up. Killing all background processes..."
    kill $(jobs -p) 2>/dev/null || true
    exit 1
}

# Set up interrupt and error handlers
trap cleanup SIGINT SIGTERM ERR EXIT

# Process each subfolder
for subfolder in "${SUBFOLDERS[@]}"; do
    echo "Processing subfolder: $subfolder"
    
    # Determine which videos to process
    # Check if specific videos are defined in config
    SPECIFIC_VIDEOS_RAW=$(./parse_yaml.py "$CONFIG_FILE" dataset.specific_videos 2>/dev/null || echo "")
    
    if [[ "$SPECIFIC_VIDEOS_RAW" != "[]" && -n "$SPECIFIC_VIDEOS_RAW" ]]; then
        echo "Using specific videos from config: ${SPECIFIC_VIDEOS[*]}"
        # Create a verified list of videos that exist
        valid_videos=()
        for video_name in "${SPECIFIC_VIDEOS[@]}"; do
            # Check if the transformed video directory exists
            if [ -d "$EXPERIMENT_DIR/$subfolder/$video_name" ]; then
                valid_videos+=("$video_name")
            else
                echo "Warning: Specified video directory not found: $EXPERIMENT_DIR/$subfolder/$video_name"
            fi
        done
        videos=("${valid_videos[@]}")
    else
        # Find all video directories in the experiment directory
        videos=()
        for dir in "$EXPERIMENT_DIR/$subfolder"/*; do
            if [ -d "$dir" ]; then
                video_name=$(basename "$dir")
                videos+=("$video_name")
            fi
        done
    fi
    
    echo "Found ${#videos[@]} videos to process in subfolder $subfolder: ${videos[*]}"
    
    # Process each video
    for video_name in "${videos[@]}"; do
        echo "Processing video: $video_name"
        
        # Get transformed video path (render.mp4)
        video_path="$EXPERIMENT_DIR/$subfolder/$video_name/render.mp4"
        if [ ! -f "$video_path" ]; then
            echo "Warning: Rendered video file $video_path not found, skipping."
            continue
        fi
        
        # Read prompt from prompt file
        prompt_file="$PROMPTS_DIR/$video_name.txt"
        if [ ! -f "$prompt_file" ]; then
            echo "Error: Prompt file $prompt_file not found, skipping."
            continue
        fi
        
        prompt=$(cat "$prompt_file")
        
        # For each inference step
        for steps in "${NUM_INFERENCE_STEPS[@]}"; do
            # Determine output path using the template
            inverted_latent_path="$INVERSION_DIR/${subfolder}_latents_${video_name}_${steps}.pt"
            
            # Check if inverted latent already exists
            if [ -f "$inverted_latent_path" ]; then
                echo "Inverted latent already exists: $inverted_latent_path, skipping..."
                continue
            fi
            
            # Get GPU ID for this iteration
            GPU_IDX=$((COUNTER % NUM_GPUS))
            GPU_ID=${GPU_DEVICE_IDS[$GPU_IDX]}
            
            echo "Running inversion for $video_name on GPU $GPU_ID with $steps steps..."
            
            # Run the inversion script in background
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            CUDA_VISIBLE_DEVICES=$GPU_ID python i2v_ddim_inversion.py \
                --model_path "$MODEL_PATH" \
                --lora_path "$LORA_PATH" \
                --prompt "$prompt" \
                --video_path "$video_path" \
                --output_path "$INVERSION_DIR" \
                --subfolder "$subfolder" \
                --num_inference_steps "$steps" \
                --seed $SEED || {
                    echo "Error: inversion failed for $video_name with exit code $?"
                    cleanup
                }
            
            # No need to store PID or run in background anymore
            # We'll process one at a time to ensure reliability
        done
    done
done

# If we got here, all processes completed successfully
echo "All inversions completed successfully!"
# Reset the trap before exiting normally
trap - ERR EXIT SIGINT SIGTERM
exit 0 