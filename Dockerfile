FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

RUN apt install nano

# create a non-root user
#ARG USER_ID=1000
#RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo


# Arguments for user/group details. These are defaults.
# The actual UID/GID will be passed at runtime.
ARG DOCKER_UID=1000
ARG DOCKER_GID=1000


# Create the group and user
# Use -r for system user/group if needed, but skipping it here for clarity
# Use --no-create-home unless a home directory is required
RUN groupadd -g ${DOCKER_GID} appgroup && \
    useradd -u ${DOCKER_UID} -g appgroup -m -s /bin/bash appuser



RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install tensorboard cmake onnx   # cmake from apt-get is too old
RUN pip install torch==2.1.0+cu121 torchvision==0.16 -f https://download.pytorch.org/whl/cu121/torch_stable.html

USER appuser
RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN pip install git+https://github.com/cocodataset/panopticapi.git

# for tensorboard (newer, tested on 5.29, causes error, but <=3.20.1 is too old)
RUN pip install protobuf==3.20.2

WORKDIR /home/appuser