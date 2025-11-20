#!/bin/bash

# Configuration
REPO_URL="https://github.com/google-ai-edge/LiteRT.git"
# Path where the repo will be temporarily cloned
CLONE_PATH="LiteRT_temp_clone"
# Path where the folders will be copied to (e.g. "./libs", "./external", or ".")
OUTPUT_DIR="../../../"

# 1. Clone the repository to a temporary directory
echo "Cloning LiteRT repository..."
# Remove existing temp dir if it exists to ensure a clean clone
rm -rf "$CLONE_PATH"
git clone "$REPO_URL" "$CLONE_PATH"

# Check if clone was successful
if [ ! -d "$CLONE_PATH" ]; then
    echo "Error: Failed to clone repository."
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 2. Copy the specific folders to the output directory
echo "Copying litert, tflite, and third_party folders..."

if [ -d "$CLONE_PATH/litert" ]; then
    cp -r "$CLONE_PATH/litert" "$OUTPUT_DIR/"
else
    echo "Warning: 'litert' folder not found in repo."
fi

if [ -d "$CLONE_PATH/tflite" ]; then
    cp -r "$CLONE_PATH/tflite" "$OUTPUT_DIR/"
else
    echo "Warning: 'tflite' folder not found in repo."
fi

if [ -d "$CLONE_PATH/third_party" ]; then
    cp -r "$CLONE_PATH/third_party" "$OUTPUT_DIR/"
else
    echo "Warning: 'third_party' folder not found in repo."
fi

# 3. Delete the cloned repository
echo "Cleaning up..."
rm -rf "$CLONE_PATH"

echo "Success! Dependencies fetched and cleanup complete."