#!/bin/bash

set -e  # Exit on error

VENV_PATH="../inverse-dvs-env"
VENV_PYTHON="$VENV_PATH/bin/python"
VENV_PIP="$VENV_PATH/bin/pip"

# Check if python3.10 exists
if ! command -v python3.10 &> /dev/null; then
    echo "Error: python3.10 not found. Please install python3.10 first."
    exit 1
fi

# Create virtual environment in /mnt using python3.10
echo "Creating virtual environment at $VENV_PATH..."
python3.10 -m venv "$VENV_PATH"

# Check if venv was created successfully
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Failed to create virtual environment at $VENV_PATH"
    exit 1
fi

# Upgrade pip using venv's pip directly
echo "Upgrading pip..."
"$VENV_PIP" install --upgrade pip

# Install requirements using venv's pip directly
echo "Installing requirements from requirements.txt..."
"$VENV_PIP" install -r requirements.txt

# Fix the import in augmentations.py
AUGMENTATIONS_FILE="$VENV_PATH/lib/python3.10/site-packages/pytorchvideo/transforms/augmentations.py"
if [ -f "$AUGMENTATIONS_FILE" ]; then
    echo "Fixing import in augmentations.py..."
    sed -i 's/import torchvision.transforms.functional_tensor as F_t/import torchvision.transforms.functional as F_t/g' "$AUGMENTATIONS_FILE"
    echo "Import fixed successfully!"
else
    echo "Warning: augmentations.py not found at $AUGMENTATIONS_FILE"
    echo "This might happen if pytorchvideo is not installed yet."
fi

if [ ! -d "Video-Depth-Anything/checkpoints" ]; then
    cd Video-Depth-Anything
    bash get_weights.sh
    cd ..
fi

echo "Setup complete!"
echo "To activate the environment, run: source $VENV_PATH/bin/activate"