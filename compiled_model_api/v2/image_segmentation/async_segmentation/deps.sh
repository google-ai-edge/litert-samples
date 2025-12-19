#!/bin/bash
#
# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Script to setup dependency files for the image segmentation c++ app example

# --- Configuration ---
REPO_URL="https://github.com/google-ai-edge/LiteRT.git"

# The specific commit hash to clone (Pinned for stability)
# You can still override this by passing an argument to the script: ./setup.sh <other_sha>
DEFAULT_SHA="fb853535a6085fb4c497514b64a4f07b555ddf11"
TARGET_COMMIT="${1:-$DEFAULT_SHA}" 

CLONE_PATH="LiteRT_temp_clone"
OUTPUT_DIR="./"

# Stop script on first error
set -e 

# --- Functions ---

cleanup() {
    if [ -d "$CLONE_PATH" ]; then
        echo "Cleaning up temporary files..."
        rm -rf "$CLONE_PATH"
    fi
}

# Ensure cleanup runs even if script is interrupted or fails
trap cleanup EXIT

# --- Main Execution ---

echo "=== LiteRT Dependency Setup ==="
echo "Targeting Commit: $TARGET_COMMIT"

# 1. Clone the repository
echo "1. Cloning LiteRT repository..."
cleanup # Ensure clean slate
# Note: We do a full clone (no --depth) to ensure we can find the specific history/SHA
git clone "$REPO_URL" "$CLONE_PATH"

# 2. Checkout the specific commit
echo "2. Checking out specific commit..."
pushd "$CLONE_PATH" > /dev/null
    git checkout "$TARGET_COMMIT"
    
    # Optional: verify we are actually on the right commit
    CURRENT_HEAD=$(git rev-parse HEAD)
    echo "   -> Verified HEAD is now at: $CURRENT_HEAD"
popd > /dev/null

# 3. Create output directory
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 4. Copy specific folders
echo "3. Copying required folders to $OUTPUT_DIR..."

FOLDERS_TO_COPY=("litert" "tflite" "third_party")

for folder in "${FOLDERS_TO_COPY[@]}"; do
    SRC="$CLONE_PATH/$folder"
    DEST="$OUTPUT_DIR/"
    
    if [ -d "$SRC" ]; then
        echo "   - Copying $folder..."
        cp -r "$SRC" "$DEST"
    else
        echo "   [!] Warning: '$folder' folder not found in commit $TARGET_COMMIT."
    fi
done

echo "=== Success! Dependencies fetched from commit $TARGET_COMMIT ==="