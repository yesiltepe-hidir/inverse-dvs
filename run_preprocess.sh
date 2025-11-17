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

# Default config file
CONFIG_FILE="config.yml"

# Allow specifying a different config file as an argument
if [ $# -eq 1 ]; then
    CONFIG_FILE="$1"
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
VIDEO_PATTERN=$(python3 parse_yaml.py "$CONFIG_FILE" "dataset.video_pattern")
DEPTHS_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "depth.output_dir")
DEPTH_ENCODER=$(python3 parse_yaml.py "$CONFIG_FILE" "depth.encoder")
SAVE_NPZ=$(python3 parse_yaml.py "$CONFIG_FILE" "depth.save_npz")
FORCE_EXTRACT_DEPTHS=$(python3 parse_yaml.py "$CONFIG_FILE" "depth.force_extract")
VDA_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "depth.video_depth_anything_dir")
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

echo "Using GPUs: ${GPU_DEVICE_IDS[@]}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Export GPU settings
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# Create necessary directories
mkdir -p "$DEPTHS_DIR"
mkdir -p "$PROMPTS_DIR"

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
    echo "No specific videos defined, processing all videos matching pattern: $VIDEO_PATTERN"
    # Find all video files matching the pattern
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
    
    echo "Processing video: $video_name"
    video_path="$DATA_ROOT_DIR/$video_name.mp4"
    
    if [ ! -f "$video_path" ]; then
        echo "Warning: Video file not found at $video_path. Skipping."
        continue
    fi
    
    # Step 1: Extract frames if needed
    # if [ "$FORCE_EXTRACT_FRAMES" = "true" ] || [ ! -d "$FRAMES_DIR/$video_name" ]; then
    #     echo "Extracting frames for $video_name..."
    #     python utils/video2frames.py --video_root_path "$DATA_ROOT_DIR"
    # else
    #     echo "Frames already exist for $video_name and force_extract is false. Skipping frame extraction."
    # fi
    
    # Step 2: Extract depths if needed
    if [ "$FORCE_EXTRACT_DEPTHS" = "true" ] || [ ! -f "$DEPTHS_DIR/${video_name}/${video_name}_depths.npz" ]; then
        echo "Extracting depths for $video_name..."
        depth_output_dir="$DEPTHS_DIR/$video_name"
        mkdir -p "$depth_output_dir"
        
        # Save current directory to return later
        CURRENT_DIR=$(pwd)
        
        # Change to Video-Depth-Anything directory
        cd "$VDA_DIR"
        
        # Build save_npz flag if enabled
        SAVE_NPZ_FLAG=""
        if [ "$SAVE_NPZ" ]; then
            SAVE_NPZ_FLAG="--save_npz"
        fi
        
        # Run depth extraction
        python3 run.py --input_video "../$video_path" --output_dir "../$depth_output_dir" --encoder "$DEPTH_ENCODER" $SAVE_NPZ_FLAG
        
        # Return to original directory
        cd "$CURRENT_DIR"
    else
        echo "Depths already exist for $video_name and force_extract is false. Skipping depth extraction."
    fi
    
    echo "Completed processing for $video_name"
    echo "-----------------------------------"
    
    # Check for 'q' input after each video
    check_for_quit
done

echo "Preprocessing complete!" 