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

docker build -t arctururs-dipr-m1:dev -f m1-docker/Dockerfile .

DATA_DIR=`realpath "$1"`
echo "Mounting $DATA_DIR to /data in the container"
docker run --mount type=bind,source="$DATA_DIR",target=/data arctururs-dipr-m1:dev python3 /dipr/dipr/evaluate.py --data_folder /data
