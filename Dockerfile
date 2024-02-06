#########################################
# Create CUDA/Torch env and install PCDet
# From https://github.com/open-mmlab/OpenPCDet/blob/master/docker/cu116.Dockerfile
#########################################

# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Set environment variables
ENV NVENCODE_CFLAGS "-I/usr/local/cuda/include"
ENV CV_VERSION=4.2.0
ENV DEBIAN_FRONTEND=noninteractive

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip unzip libssl-dev libcairo2-dev lsb-release libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev software-properties-common \
    build-essential cmake pkg-config libapr1-dev autoconf automake libtool curl libc6 libboost-all-dev debconf libomp5 libstdc++6 \
    libqt5core5a libqt5xml5 libqt5gui5 libqt5widgets5 libqt5concurrent5 libqt5opengl5 libcap2 libusb-1.0-0 libatk-adaptor neovim \
    python3-pip python3-tornado python3-dev python3-numpy python3-virtualenv libpcl-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libsuitesparse-dev python3-pcl pcl-tools libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libdc1394-22-dev xfce4-terminal &&\
    rm -rf /var/lib/apt/lists/*

# OpenCV with CUDA support
# WORKDIR /opencv
# RUN git clone https://github.com/opencv/opencv.git -b $CV_VERSION &&\
#     git clone https://github.com/opencv/opencv_contrib.git -b $CV_VERSION

# # While using OpenCV 4.2.0 we have to apply some fixes to ensure that CUDA is fully supported, thanks @https://github.com/gismo07 for this fix
# RUN mkdir opencvfix && cd opencvfix &&\
#     git clone https://github.com/opencv/opencv.git -b 4.5.2 &&\
#     cd opencv/cmake &&\
#     cp -r FindCUDA /opencv/opencv/cmake/ &&\
#     cp FindCUDA.cmake /opencv/opencv/cmake/ &&\
#     cp FindCUDNN.cmake /opencv/opencv/cmake/ &&\
#     cp OpenCVDetectCUDA.cmake /opencv/opencv/cmake/
 
# WORKDIR /opencv/opencv/build

# RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
# -D CMAKE_INSTALL_PREFIX=/usr/local \
# -D OPENCV_GENERATE_PKGCONFIG=ON \
# -D BUILD_EXAMPLES=OFF \
# -D INSTALL_PYTHON_EXAMPLES=OFF \
# -D INSTALL_C_EXAMPLES=OFF \
# -D PYTHON_EXECUTABLE=$(which python2) \
# -D PYTHON3_EXECUTABLE=$(which python3) \
# -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
# -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
# -D BUILD_opencv_python2=ON \
# -D BUILD_opencv_python3=ON \
# -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
# -D WITH_GSTREAMER=ON \
# -D WITH_CUDA=ON \
# -D ENABLE_PRECOMPILED_HEADERS=OFF \
# .. &&\
# make -j$(nproc) &&\
# make install &&\
# ldconfig &&\
# rm -rf /opencv

# WORKDIR /
# ENV OpenCV_DIR=/usr/share/OpenCV

# PyTorch for CUDA 11.6
RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    
# OpenPCDet
RUN pip3 install numpy==1.23.0 llvmlite numba tensorboardX easydict pyyaml scikit-image tqdm SharedArray open3d mayavi av2 kornia==0.5.8 pyquaternion
RUN pip3 install spconv-cu116

RUN git clone https://github.com/open-mmlab/OpenPCDet.git

WORKDIR OpenPCDet

RUN python3 setup.py develop
    
WORKDIR /

ENV NVIDIA_VISIBLE_DEVICES="all" \
    OpenCV_DIR=/usr/share/OpenCV \
    NVIDIA_DRIVER_CAPABILITIES="video,compute,utility,graphics" \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib \
    QT_GRAPHICSSYSTEM="native"
















#######################
# Install ROS2 Galactic
#######################

# RUN echo 'Etc/UTC' > /etc/timezone && \
#     ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
#     apt-get update && \
#     apt-get install -q -y --no-install-recommends tzdata && \
#     rm -rf /var/lib/apt/lists/*

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











#####################
# Install detector pre-requisites
# RUN apt-get update \
#     && apt-get install -y pip git \
#     # ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 lsb-core \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*


# COPY ./requirements.txt requirements.txt
# RUN python3 -m pip uninstall numpy
# RUN python3 -m pip uninstall scikit-image
# RUN python3 -m pip install -r requirements.txt
# RUN python3 -m pip install cumm-cu113 spconv-cu113 wandb
# RUN python3 -m pip install torch==1.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# RUN python3 -m pip install -U scipy


# WORKDIR /

# ENV NVIDIA_VISIBLE_DEVICES="all" \
#     OpenCV_DIR=/usr/share/OpenCV \
#     NVIDIA_DRIVER_CAPABILITIES="video,compute,utility,graphics" \
#     LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib \
#     QT_GRAPHICSSYSTEM="native"


#####################
# Install CODa Models
# RUN conda clean --all

COPY . /coda-models
WORKDIR /coda-models
# RUN python3 setup.py develop


# Dataset specific installation
# RUN pip install nuscenes-devkit==1.0.5 
# RUN pip install waymo-open-dataset-tf-2-5-0 --user

COPY ./coda_entrypoint.sh /coda_entrypoint.sh
RUN chmod +x /coda_entrypoint.sh
