FROM nvcr.io/nvidia/isaac-lab:2.1.0
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    vim \
    git \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]