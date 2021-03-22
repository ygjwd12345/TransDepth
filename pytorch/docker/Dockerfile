FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5"

RUN apt-get clean && apt-get update && apt-get install -y \
    tree \
    vim \
    python3-tk \
    git \
    unzip \
    wget \
    screen \
    libgl1-mesa-dev\
    libgtk2.0-dev
RUN pip install --no-cache-dir Cython \
    wandb \
    torchvision \
    ninja \
    scikit-image \
    opencv-python \
    pandas \
    tqdm \
    scikit-learn \
    inplace_abn \
    tensorboard \
    tensorboardX
RUN pip install pycocotools \
    opencv-contrib-python
WORKDIR "/"
RUN pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
WORKDIR "/home"