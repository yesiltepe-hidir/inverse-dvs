#!/bin/bash

# Cleanup function to make sure we terminate all child processes
cleanup() {
    local exit_status=$?
    echo "Cleaning up and stopping all processes. Exit status: $exit_status"
    pkill -P $$ # Kill all child processes
    exit $exit_status
}

# Set up traps for various signals and errors
trap cleanup EXIT
trap 'cleanup' INT TERM

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

# Check that all required scripts exist
check_script "./run_preprocess.sh"
check_script "./run_camera.sh"
check_script "./run_inversion.sh"
check_script "./run_infer.sh"

# Display GPU information
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

CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_DEVICE_IDS[*]}")
echo "Using GPUs: ${GPU_DEVICE_IDS[@]}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Track total pipeline start time
PIPELINE_START_TIME=$(date +%s)

# First, run preprocessing (only needed once for all subfolders)
echo "============================================================"
echo "STEP 1: Running Preprocessing"
echo "============================================================"
PREPROCESS_START_TIME=$(date +%s)

# Run preprocessing
./run_preprocess.sh "$CONFIG_FILE"
PREPROCESS_RESULT=$?

if [ $PREPROCESS_RESULT -ne 0 ]; then
    echo "Error: Preprocessing failed with exit code $PREPROCESS_RESULT"
    exit $PREPROCESS_RESULT
fi

PREPROCESS_END_TIME=$(date +%s)
PREPROCESS_DURATION=$((PREPROCESS_END_TIME - PREPROCESS_START_TIME))
echo "Preprocessing completed in $PREPROCESS_DURATION seconds"

# Get subfolders from config file
readarray -t SUBFOLDERS < <(python3 parse_yaml.py "$CONFIG_FILE" "experiment.trajectories[]")

if [ ${#SUBFOLDERS[@]} -eq 0 ]; then
    echo "Error: No subfolders defined in config file under experiment.trajectories"
    exit 1
fi

echo "Found ${#SUBFOLDERS[@]} subfolder(s) to process: ${SUBFOLDERS[*]}"

# Initialize timing arrays
declare -a CAMERA_DURATIONS
declare -a INVERSION_DURATIONS
declare -a INFERENCE_DURATIONS

# Process each subfolder
for ((i=0; i<${#SUBFOLDERS[@]}; i++)); do
    SUBFOLDER="${SUBFOLDERS[$i]}"
    echo ""
    echo "============================================================"
    echo "PROCESSING SUBFOLDER ($((i+1))/${#SUBFOLDERS[@]}): $SUBFOLDER"
    echo "============================================================"
    
    # Track subfolder start time
    SUBFOLDER_START_TIME=$(date +%s)
    
    # Run camera transformation
    echo "============================================================"
    echo "STEP 2: Running Camera Transformation for subfolder: $SUBFOLDER"
    echo "============================================================"
    CAMERA_START_TIME=$(date +%s)
    
    ./run_camera.sh "$SUBFOLDER" "$CONFIG_FILE"
    CAMERA_RESULT=$?
    
    if [ $CAMERA_RESULT -ne 0 ]; then
        echo "Error: Camera transformation failed for subfolder $SUBFOLDER with exit code $CAMERA_RESULT"
        exit $CAMERA_RESULT
    fi
    
    CAMERA_END_TIME=$(date +%s)
    CAMERA_DURATION=$((CAMERA_END_TIME - CAMERA_START_TIME))
    CAMERA_DURATIONS[$i]=$CAMERA_DURATION
    echo "Camera transformation completed in $CAMERA_DURATION seconds"
    
    # Run inversion
    echo "============================================================"
    echo "STEP 3: Running Inversion for subfolder: $SUBFOLDER"
    echo "============================================================"
    INVERSION_START_TIME=$(date +%s)
    
    ./run_inversion.sh "$SUBFOLDER" "$CONFIG_FILE"
    INVERSION_RESULT=$?
    
    if [ $INVERSION_RESULT -ne 0 ]; then
        echo "Error: Inversion failed for subfolder $SUBFOLDER with exit code $INVERSION_RESULT"
        # Make sure we exit cleanly
        exit $INVERSION_RESULT
    fi
    
    INVERSION_END_TIME=$(date +%s)
    INVERSION_DURATION=$((INVERSION_END_TIME - INVERSION_START_TIME))
    INVERSION_DURATIONS[$i]=$INVERSION_DURATION
    echo "Inversion completed in $INVERSION_DURATION seconds"
    
    # Run inference
    echo "============================================================"
    echo "STEP 4: Running Inference for subfolder: $SUBFOLDER"
    echo "============================================================"
    INFERENCE_START_TIME=$(date +%s)
    
    ./run_infer.sh "$SUBFOLDER" "$CONFIG_FILE"
    INFERENCE_RESULT=$?
    
    if [ $INFERENCE_RESULT -ne 0 ]; then
        echo "Error: Inference failed for subfolder $SUBFOLDER with exit code $INFERENCE_RESULT"
        exit $INFERENCE_RESULT
    fi
    
    INFERENCE_END_TIME=$(date +%s)
    INFERENCE_DURATION=$((INFERENCE_END_TIME - INFERENCE_START_TIME))
    INFERENCE_DURATIONS[$i]=$INFERENCE_DURATION
    echo "Inference completed in $INFERENCE_DURATION seconds"
    
    # Calculate subfolder duration
    SUBFOLDER_END_TIME=$(date +%s)
    SUBFOLDER_DURATION=$((SUBFOLDER_END_TIME - SUBFOLDER_START_TIME))
    echo "============================================================"
    echo "Subfolder $SUBFOLDER completed in $SUBFOLDER_DURATION seconds"
    echo "============================================================"
    echo ""
done

# Calculate and display total duration
PIPELINE_END_TIME=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo "============================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "============================================================"
echo "Total execution time: $PIPELINE_DURATION seconds"
echo "  - Preprocessing: $PREPROCESS_DURATION seconds"
echo ""

# Print timing information for each subfolder
for ((i=0; i<${#SUBFOLDERS[@]}; i++)); do
    SUBFOLDER="${SUBFOLDERS[$i]}"
    echo "Subfolder: $SUBFOLDER"
    echo "  - Camera transformation: ${CAMERA_DURATIONS[$i]} seconds"
    echo "  - Inversion: ${INVERSION_DURATIONS[$i]} seconds"
    echo "  - Inference: ${INFERENCE_DURATIONS[$i]} seconds"
    echo ""
done

echo "Configuration: $CONFIG_FILE"
echo "============================================================" 