#!/bin/sh

TARGET_DIR="datasets"
REPO_URL="https://huggingface.co/datasets/qwedsacf/competition_math"
REPO_NAME="competition_math"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating $TARGET_DIR directory..."
    mkdir -p "$TARGET_DIR"
fi

if [ -d "$TARGET_DIR/$REPO_NAME" ]; then
    echo "Dataset $REPO_NAME already exists in $TARGET_DIR. Updating..."
    cd "$TARGET_DIR/$REPO_NAME" && git pull
else
    echo "Downloading $REPO_NAME to $TARGET_DIR..."
    cd "$TARGET_DIR" && git clone "$REPO_URL"
fi
