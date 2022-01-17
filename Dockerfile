FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=nonintercative
RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev  ffmpeg libsm6 libxext6 && \
        rm -rf /var/lib/apt/lists/*
RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --upgrade setuptools
COPY . /dipr
RUN cd /dipr && pip3 install .
