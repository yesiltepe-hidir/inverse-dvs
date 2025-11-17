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

# Function to check script existence
check_script() {
    local script="$1"
    if [ ! -f "$script" ]; then
        echo "Error: Script $script not found!"
        exit 1
    fi
    
    # Ensure script is executable
    if [ ! -x "$script" ]; then
        echo "Making $script executable..."
        chmod +x "$script"
    fi
}

# Check for the inversion script
check_script "./run_inversion.sh"

# Parse configuration
DATA_ROOT_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "dataset.root_dir")
MODEL_PATH=$(python3 parse_yaml.py "$CONFIG_FILE" "inference.model_path")
LORA_PATH=$(python3 parse_yaml.py "$CONFIG_FILE" "inference.lora_path")
EXPERIMENT_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "common.experiment_dir")
PROMPTS_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "common.prompts_dir")
INVERSION_DIR=$(python3 parse_yaml.py "$CONFIG_FILE" "inference.inversion_dir")
INVERTED_LATENT_TEMPLATE=$(python3 parse_yaml.py "$CONFIG_FILE" "inference.inverted_latent_template")
MASK_PATH_TEMPLATE=$(python3 parse_yaml.py "$CONFIG_FILE" "inference.mask_path_template")
DEPTH_PATH_TEMPLATE=$(python3 parse_yaml.py "$CONFIG_FILE" "inference.depth_path_template")
SEED=$(python3 parse_yaml.py "$CONFIG_FILE" "inference.seed")

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
NUM_GPUS=${#GPU_DEVICE_IDS[@]}

echo "Using GPUs: ${GPU_DEVICE_IDS[@]}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Parse parameter lists
readarray -t NUM_INFERENCE_STEPS_LIST < <(python3 parse_yaml.py "$CONFIG_FILE" "inference.num_inference_steps[]")
readarray -t GUIDANCE_SCALE_LIST < <(python3 parse_yaml.py "$CONFIG_FILE" "inference.guidance_scale[]")
readarray -t PRESERVATION_SCALE_LIST < <(python3 parse_yaml.py "$CONFIG_FILE" "inference.preservation_scale[]")
readarray -t K_ORDER_LIST < <(python3 parse_yaml.py "$CONFIG_FILE" "inference.k_order[]")
readarray -t TRESHOLD_INDEX_LIST < <(python3 parse_yaml.py "$CONFIG_FILE" "inference.treshold_index[]")

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

echo "Starting inference with the following parameters:"
echo "- Number of GPUs: ${#GPU_DEVICE_IDS[@]}"
echo "- Inference steps: ${NUM_INFERENCE_STEPS_LIST[@]}"
echo "- Guidance scales: ${GUIDANCE_SCALE_LIST[@]}"
echo "- Preservation scales: ${PRESERVATION_SCALE_LIST[@]}"
echo "- k orders: ${#K_ORDER_LIST[@]} values"

# Check if specific videos are defined
# First, try to get the raw specific_videos value (will be empty string if not defined or empty array)
SPECIFIC_VIDEOS_RAW=$(python3 parse_yaml.py "$CONFIG_FILE" "dataset.specific_videos" 2>/dev/null || echo "")

# Check if the specific_videos value contains something other than empty array
if [[ "$SPECIFIC_VIDEOS_RAW" != "[]" && -n "$SPECIFIC_VIDEOS_RAW" ]]; then
    echo "Processing specific videos defined in config: $SPECIFIC_VIDEOS_RAW"
    # Get array of specific videos
    readarray -t SPECIFIC_VIDEOS < <(python3 parse_yaml.py "$CONFIG_FILE" "dataset.specific_videos[]")
    
    # Verify each specific video exists
    VIDEOS=()
    for video_name in "${SPECIFIC_VIDEOS[@]}"; do
        video_path="$DATA_ROOT_DIR/$video_name.mp4"
        video_dir="$EXPERIMENT_DIR/$SUBFOLDER/$video_name"
        
        if [ -f "$video_path" ] && [ -d "$video_dir" ]; then
            VIDEOS+=("$video_name")
        else
            if [ ! -f "$video_path" ]; then
                echo "Warning: Specified video file not found: $video_path"
            fi
            if [ ! -d "$video_dir" ]; then
                echo "Warning: Specified video directory not found: $video_dir"
            fi
        fi
    done
else
    echo "No specific videos defined, processing all videos in output directory..."
    # Find all video directories in the output folder
    VIDEOS=()
    for dir in "$EXPERIMENT_DIR/$SUBFOLDER"/*; do
        if [ -d "$dir" ]; then
            video_name=$(basename "$dir")
            # Check if the actual video file exists as well
            video_path="$DATA_ROOT_DIR/$video_name.mp4"
            if [ -f "$video_path" ]; then
                VIDEOS+=("$video_name")
            else
                echo "Warning: Found directory but no video file for: $video_name"
            fi
        fi
    done
fi

echo "Found ${#VIDEOS[@]} videos to process: ${VIDEOS[*]}"

# First, check if we need to run inversion for any videos
need_inversion=false
for video_name in "${VIDEOS[@]}"; do
    for num_inference_steps in "${NUM_INFERENCE_STEPS_LIST[@]}"; do
        # Replace template variables to determine inverted latent path
        inverted_latent_path=$(echo "$INVERTED_LATENT_TEMPLATE" | sed \
            -e "s/{inversion_dir}/$INVERSION_DIR/g" \
            -e "s/{subfolder}/$SUBFOLDER/g" \
            -e "s/{video_name}/$video_name/g" \
            -e "s/{num_inference_steps}/$num_inference_steps/g")
            
        if [ ! -f "$inverted_latent_path" ]; then
            need_inversion=true
            break 2  # Break out of both loops
        fi
    done
done

# If we need inversion, run the inversion script
if [ "$need_inversion" = "true" ]; then
    echo "============================================================"
    echo "Some inverted latents are missing. Running inversion step..."
    echo "============================================================"
    
    ./run_inversion.sh "$SUBFOLDER" "$CONFIG_FILE"
    INVERSION_RESULT=$?
    
    if [ $INVERSION_RESULT -ne 0 ]; then
        echo "Error: Inversion failed with exit code $INVERSION_RESULT"
        exit $INVERSION_RESULT
    fi
    
    echo "============================================================"
    echo "Inversion completed. Continuing with inference..."
    echo "============================================================"
else
    echo "All required inverted latents exist. Skipping inversion step."
fi

# Counter for GPU allocation
COUNTER=0

# Process videos
for video_name in "${VIDEOS[@]}"; do
    video_dir="$EXPERIMENT_DIR/$SUBFOLDER/$video_name"
    
    if [ ! -d "$video_dir" ]; then
        echo "Warning: Directory not found at $video_dir. Skipping."
        continue
    fi
    
    video_path="$DATA_ROOT_DIR/$video_name.mp4"
    
    if [ ! -f "$video_path" ]; then
        echo "Warning: Video file not found at $video_path. Skipping."
        continue
    fi
    
    echo "Processing video: $video_name"
    
    # Read prompt from prompt.txt file
    prompt_file="$PROMPTS_DIR/$video_name.txt"
    if [ ! -f "$prompt_file" ]; then
        echo "Warning: Prompt file not found at $prompt_file. Skipping."
        continue
    fi
    prompt=$(cat "$prompt_file")
    echo "Prompt: $prompt"
    
    for num_inference_steps in "${NUM_INFERENCE_STEPS_LIST[@]}"; do
        # Replace template variables to determine inverted latent path
        inverted_latent_path=$(echo "$INVERTED_LATENT_TEMPLATE" | sed \
            -e "s/{inversion_dir}/$INVERSION_DIR/g" \
            -e "s/{subfolder}/$SUBFOLDER/g" \
            -e "s/{video_name}/$video_name/g" \
            -e "s/{num_inference_steps}/$num_inference_steps/g")
            
        # Double-check that inverted latent exists
        if [ ! -f "$inverted_latent_path" ]; then
            echo "Warning: Inverted latent not found at $inverted_latent_path even after inversion step. Skipping."
            continue
        fi
        
        for guidance_scale in "${GUIDANCE_SCALE_LIST[@]}"; do
            for preservation_scale in "${PRESERVATION_SCALE_LIST[@]}"; do
                for k_order in "${K_ORDER_LIST[@]}"; do
                    for treshold_idx in "${TRESHOLD_INDEX_LIST[@]}"; do
                        # Check for 'q' input
                        check_for_quit
                        
                        # Get GPU ID for this iteration
                        GPU_IDX=$((COUNTER % NUM_GPUS))
                        GPU=${GPU_DEVICE_IDS[$GPU_IDX]}
                        
                        echo "Running inference for $video_name on GPU $GPU"
                        echo "- num_inference_steps: $num_inference_steps"
                        echo "- guidance_scale: $guidance_scale"
                        echo "- preservation_scale: $preservation_scale"
                        echo "- k_order: $k_order"
                        echo "- treshold_idx: $treshold_idx"
                        
                        # Replace template variables for mask path
                        mask_path=$(echo "$MASK_PATH_TEMPLATE" | sed \
                            -e "s|{experiment_dir}|$EXPERIMENT_DIR|g" \
                            -e "s|{subfolder}|$SUBFOLDER|g" \
                            -e "s|{video_name}|$video_name|g")
                        
                        # Replace template variables for depth path
                        depth_path=$(echo "$DEPTH_PATH_TEMPLATE" | sed \
                            -e "s|{video_name}|$video_name|g")
                        
                        output_path="$EXPERIMENT_DIR/$SUBFOLDER/$video_name"
                        
                        # Check if output file already exists
                        output_file="$output_path/output_t:${num_inference_steps}_g:${guidance_scale}_p:${preservation_scale}_k:${k_order}.mp4"
                        if [ -f "$output_file" ]; then
                            echo "Output file already exists at $output_file. Skipping."
                            continue
                        fi
                        
                        # Run inference
                        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
                        CUDA_VISIBLE_DEVICES=$GPU python infer.py \
                            --model_path "$MODEL_PATH" \
                            --prompt "$prompt" \
                            --video_path "$video_path" \
                            --output_path "$output_path" \
                            --lora_path "$LORA_PATH" \
                            --guidance_scale $guidance_scale \
                            --mask_path "$mask_path" \
                            --depth_path "$depth_path" \
                            --inverted_latent_path "$inverted_latent_path" \
                            --k_order $k_order \
                            --num_inference_steps $num_inference_steps \
                            --preservation_scale $preservation_scale \
                            --treshold_idx $treshold_idx \
                            --seed $SEED &
                        
                        # Increment counter
                        ((COUNTER++))
                        
                        # If we've started processes on all GPUs, wait for them to complete
                        if [ $((COUNTER % NUM_GPUS)) -eq 0 ]; then
                            wait
                            # Check for quit signal after batch completion
                            check_for_quit
                        fi
                    done
                done
            done
        done
    done
done

# Wait for all background processes to complete
wait

echo "All inference completed successfully!" 