ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS pytorch
# FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:11.0.3-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list \
#     && rm /etc/apt/sources.list.d/nvidia-ml.list \
#     && apt-key del 7fa2af80 \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# Install the required packages
RUN apt-get update \
    && apt-get install -y pip ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 lsb-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Minimal setup
RUN apt-get update && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

#######################
# Install ROS2 Galactic
#######################

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
# RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA
RUN apt update && apt install -y curl
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# setup sources.list
# RUN echo "deb http://snapshots.ros.org/galactic/final/ubuntu focal main" > /etc/apt/sources.list.d/ros2-snapshots.list
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO galactic

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-galactic-ros-core \
    ros-galactic-sensor-msgs \
    ros-galactic-sensor-msgs-py \
    ros-galactic-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

# Install coda-models
# RUN conda clean --all
COPY . /coda-models
WORKDIR /coda-models

ENV FORCE_CUDA="1"
# RUN python3 -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --index-url https://download.pytorch.org/whl/cu116
# RUN python3 -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 --index-url https://download.pytorch.org/whl/cu110
RUN python3 -m pip install torch==1.10.0+cu113 --index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install -r requirements.txt
# RUN python3 -m pip install cumm-cu113 spconv-cu113 wandb


# RUN python3 setup.py develop --user


# Add below to resolve binary incompatibility issues
# RUN pip install numpy==1.20.1

# Dataset specific installation
# RUN pip install nuscenes-devkit==1.0.5 
# RUN pip install waymo-open-dataset-tf-2-5-0 --user

