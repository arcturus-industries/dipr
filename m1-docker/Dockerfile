FROM python:3.8-slim

RUN apt update \
        && apt install -y bash ca-certificates libffi-dev libssl-dev xz-utils zlib1g-dev liblzma-dev \
        python3-opencv \
        && rm -rf /var/lib/apt/lists/*
        # && rm -rf /root/.cache/* \
        # && rm -rf /root/.config/pip

RUN python3 -m pip install numpy torch -f https://torch.kmtea.eu/whl/stable.html -f https://ext.kmtea.eu/whl/stable.html \
        && rm -rf /root/.cache/* \
        && rm -rf /root/.config/pip
COPY . /dipr
RUN cd /dipr && pip3 install .
