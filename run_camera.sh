#!/bin/bash

# Check for required tools
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required. Please install Python 3."
    exit 1
fi

# Ensure the YAML parser is available
if [ ! -f "parse_yaml.py" ]; then
    echo "Error: parse_yaml.py not found!"
    exit 1
fi

# Make parse_yaml.py executable
chmod +x parse_yaml.py

# Check if subfolder is provided
if [ $# -lt 1 ]; then
    echo "Error: Subfolder argument is required"
    echo "Usage: $0 <subfolder> [config_file]"
    exit 1
fi

# Get subfolder from command line argument
SUBFOLDER="$1"
echo "Subfolder: $SUBFOLDER"

# Default config file
CONFIG_FILE="config.yml"

# Allow specifying a different config file as an argument
if [ $# -eq 2 ]; then
    CONFIG_FILE="$2"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found!"
    exit 1
fi

echo "Using configuration file: $CONFIG_FILE"

# Set up trap to catch interrupt
trap_ctrlc() {
    echo "Stopping all processes. Please wait..."
    pkill -P $$ # Kill all child processes
    exit 1
}

# Function to check for 'q' input
check_for_quit() {
    if read -t 0; then
        read -n 1 input
        if [[ $input == "q" ]]; then
            echo "Quit requested. Stopping all processes..."
            pkill -P $$ # Kill all child processes
            exit 0
        fi
    fi
}

# Set up the trap for Ctrl+C
trap trap_ctrlc INT

# Parse configuration
DATA_ROOT_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "dataset.root_dir")
EXPERIMENT_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "common.experiment_dir")
VIDEO_LENGTH=$(python3 parse_yaml.py "$CONFIG_FILE" "camera.video_length")
DEPTH_PATH_TEMPLATE=$(python3 parse_yaml.py "$CONFIG_FILE" "camera.depth_path_template")
TRAJ_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "camera.traj_dir")
TRAJ_FILE_TEMPLATE=$(python3 parse_yaml.py "$CONFIG_FILE" "camera.traj_file_template")
PROMPTS_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "common.prompts_dir")

# Get the list of GPUs to use
readarray -t CONFIG_GPU_IDS < <(python3 parse_yaml.py "$CONFIG_FILE" "gpu.device_ids[]")

# Check if special value [-1] is provided (use all available GPUs)
if [ "${#CONFIG_GPU_IDS[@]}" -eq 1 ] && [ "${CONFIG_GPU_IDS[0]}" -eq -1 ]; then
    echo "Config specifies [-1] for GPU IDs: finding available GPUs..."
    # Get available GPUs using our utility
    readarray -t GPU_DEVICE_IDS < <(python3 utils/check_gpus.py -1)
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

# Convert array to comma-separated string for CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_DEVICE_IDS[*]}")

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo "Using GPUs: ${GPU_DEVICE_IDS[@]}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Create output directory if it doesn't exist
mkdir -p "$EXPERIMENT_DIR/$SUBFOLDER"
# Create prompts directory structure
mkdir -p "$PROMPTS_DIR"

# Get the number of available GPUs
NUM_GPUS=${#GPU_DEVICE_IDS[@]}
COUNTER=0

# Check if specific videos are defined
# First, try to get the raw specific_videos value (will be empty string if not defined or empty array)
SPECIFIC_VIDEOS_RAW=$(python3 parse_yaml.py "$CONFIG_FILE" "dataset.specific_videos" 2>/dev/null || echo "")

# Check if the specific_videos value contains something other than empty array
if [[ "$SPECIFIC_VIDEOS_RAW" != "[]" && -n "$SPECIFIC_VIDEOS_RAW" ]]; then
    echo "Processing specific videos defined in config: $SPECIFIC_VIDEOS_RAW"
    # Get array of specific videos
    readarray -t VIDEOS < <(python3 parse_yaml.py "$CONFIG_FILE" "dataset.specific_videos[]")
    
    # Verify each specific video exists
    VALID_VIDEOS=()
    for video_name in "${VIDEOS[@]}"; do
        video_path="$DATA_ROOT_DIR/$video_name.mp4"
        if [ -f "$video_path" ]; then
            VALID_VIDEOS+=("$video_name")
        else
            echo "Warning: Specified video not found: $video_path"
        fi
    done
    
    # Update VIDEOS array with only valid videos
    VIDEOS=("${VALID_VIDEOS[@]}")
else
    echo "No specific videos defined, processing all videos matching pattern..."
    # Find all video files matching the pattern
    VIDEO_PATTERN=$(python3 parse_yaml.py "$CONFIG_FILE" "dataset.video_pattern")
    VIDEOS=()
    while IFS= read -r -d '' video_path; do
        video_name=$(basename "$video_path" .mp4)
        VIDEOS+=("$video_name")
    done < <(find "$DATA_ROOT_DIR" -name "$VIDEO_PATTERN" -print0)
fi

echo "Found ${#VIDEOS[@]} videos to process: ${VIDEOS[*]}"

# Process each video
for video_name in "${VIDEOS[@]}"; do
    # Check for 'q' input
    check_for_quit
    
    # Get GPU ID for this iteration
    GPU_IDX=$((COUNTER % NUM_GPUS))
    GPU=${GPU_DEVICE_IDS[$GPU_IDX]}
    
    echo "Processing video: $video_name on GPU $GPU"
    
    video_path="$DATA_ROOT_DIR/$video_name.mp4"
    
    # Skip if video file doesn't exist
    if [ ! -f "$video_path" ]; then
        echo "Warning: Video file not found at $video_path. Skipping."
        continue
    fi

    # Create output subdirectory
    mkdir -p "$EXPERIMENT_DIR/$SUBFOLDER/$video_name"
    
    # Check if render.mp4 already exists (optional skip)
    if [ -f "$EXPERIMENT_DIR/$SUBFOLDER/$video_name/render.mp4" ]; then
        echo "Render.mp4 already exists for $video_name"
    else
        # Copy the input video to output directory
        cp "$video_path" "$EXPERIMENT_DIR/$SUBFOLDER/$video_name/input.mp4"
        
        # Replace template variables in the depth path
        depth_path=$(echo "$DEPTH_PATH_TEMPLATE" | sed "s/{video_name}/$video_name/g")
        
        # Replace template variables in the trajectory file path
        traj_file=$(echo "$TRAJ_FILE_TEMPLATE" | sed "s/{subfolder}/$SUBFOLDER/g")
        traj_path="$TRAJ_DIR/$traj_file"
        
        # Run warper_utils.py on the video in background on assigned GPU
        echo "Running camera transformation with depth path: $depth_path and trajectory: $traj_path"
        CUDA_VISIBLE_DEVICES=$GPU python warper_utils.py \
            --video_path "$video_path" \
            --save_dir "$EXPERIMENT_DIR/$SUBFOLDER/$video_name" \
            --video_length "$VIDEO_LENGTH" \
            --depth_path "$depth_path" \
            --traj_txt "$traj_path" &
            
        # Increment counter
        ((COUNTER++))
        
        # If we've started processes on all GPUs, wait for them to complete
        if [ $((COUNTER % NUM_GPUS)) -eq 0 ]; then
            wait
            # Check for quit signal after batch completion
            check_for_quit
        fi
        
        # Wait for this specific render.mp4 to be created
        while [ ! -f "$EXPERIMENT_DIR/$SUBFOLDER/$video_name/render.mp4" ]; do
            echo "Waiting for render.mp4 to be created for $video_name..."
            sleep 5
        done
    fi
    
    # Generate prompts if needed - using the rendered video
    prompt_file="$PROMPTS_DIR/$video_name.txt"
    render_video_path="$EXPERIMENT_DIR/$SUBFOLDER/$video_name/control_input.mp4"
    
    if [ ! -f "$prompt_file" ] && [ -f "$render_video_path" ]; then
        echo "Generating prompt for $video_name using CogVLM..."
        python cogvlm.py --video_path "$render_video_path" --prompts_dir "$PROMPTS_DIR"
    elif [ ! -f "$prompt_file" ]; then
        echo "Warning: Cannot generate prompt for $video_name as render.mp4 doesn't exist"
    else
        echo "Prompt already exists for $video_name/$SUBFOLDER. Skipping prompt generation."
    fi
    
    # Copy prompt file to output directory
    if [ -f "$prompt_file" ]; then
        cp "$prompt_file" "$EXPERIMENT_DIR/$SUBFOLDER/$video_name/prompt.txt"
    else
        echo "Warning: Prompt file not found at $prompt_file"
    fi
done

# Wait for any remaining processes
wait
echo "All processing complete!" 