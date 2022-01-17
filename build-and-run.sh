#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <data_path>"
    exit 1
fi

if [ ! $(command -v realpath) ]; then
    echo "realpath command not found. Please install it before running this script"
    echo "You can install by running"
    echo "On Linux: apt-get install coreutils"
    echo "On MacOS: brew install coreutils"
    exit 1
fi

docker build --platform linux/amd64 -t arcturus-dipr:dev .

DATA_DIR=`realpath "$1"`
echo "Mounting $DATA_DIR to /data in the container"
docker run --mount type=bind,source="$DATA_DIR",target=/data arcturus-dipr:dev python3 /dipr/dipr/evaluate.py --challenge_folder /data
