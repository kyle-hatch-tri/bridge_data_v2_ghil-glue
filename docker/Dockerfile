# -------------------------
# Base SageMaker image
# -------------------------
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker
# See https://github.com/aws/deep-learning-containers/blob/master/available_images.md

ENV PROJECT=bridgev2

# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
SHELL ["/bin/bash", "-ex", "-o", "pipefail", "-c"]

# Create an environment setup file separate from /etc/bash.bashrc, and source
# this file from /etc/bash.bashrc. The reason for this is that /etc/bash.bashrc
# doesn't execute when not running the shell interactively. If we have a separate
# file for environment setup, then we can source that file when executing
# non-interactively and still have the expected environment setup.
# This is useful for e.g., running tests with the expected environment.
# Takeaway: Instead of adding stuff to /etc/bash.bashrc, add it to this file!
ENV SHELL_SETUP_FILE=/usr/local/bin/${PROJECT}_env_setup.sh
RUN touch ${SHELL_SETUP_FILE} \
    && chmod +x ${SHELL_SETUP_FILE} \
    && echo "source ${SHELL_SETUP_FILE}" >> /etc/bash.bashrc

# -------------------------
# System tools
# -------------------------
# Configure apt for use in Docker.
RUN echo 'APT::Get::Assume-Yes "true";' | tee /etc/apt/apt.conf.d/90yes
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install --no-install-recommends -q -y \
        locales \
        openssh-client openssh-server \
        lsb-release \
        cmake \
        curl \
        docker.io \
        ffmpeg \
        git \
        htop \
        libsm6 \
        libxext6 \
        libgl1  \
        libglib2.0-0 \
        libxrender-dev \
        libxext6 \
        libosmesa6-dev \
        libglfw3 \
        libjpeg-dev \
        libpng-dev \
        patchelf \
        python3-flake8-docstrings \
        python3-pip \
        python3-pyaudio \
        python3-pytest-cov \
        sudo \
        freeglut3-dev \
        software-properties-common \
        tmux \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8
ENV \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# -------------------------
# Python packages
# -------------------------
RUN pip install --upgrade pip
# standard deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# Install cython==3.0.0 after other requirements. This conflicts with
# the requirements of robosuite, and breaks some functionality of it,
# but otherwise pip cannot solve for a configuration.
RUN pip install cython==3.0.0
# Install AWS CLI v2 (and uninstall v1). You need v2 for sagemaker sso login.
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update && \
    rm awscliv2.zip && \
    pip uninstall -y awscli && \
    aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

# -------------------------
# Project code
# -------------------------
ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
ENV WORKSPACE=/opt/ml/code
COPY . ${WORKSPACE}

WORKDIR ${WORKSPACE}
ENV PYTHONPATH="${WORKSPACE}:$PYTHONPATH"
RUN git config --global --add safe.directory ${WORKSPACE}


# Add external packages to python path.
ENV EXTERNAL="${WORKSPACE}/external"
ENV PYTHONPATH="${EXTERNAL}/dlimp:$PYTHONPATH"

RUN pip install --upgrade "jax[cuda11_pip]"==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    jaxlib==0.4.11 \
    flax==0.7.0 \
    optax==0.1.5 \
    distrax==0.1.2 \
    chex==0.1.7 

RUN pip install orbax-checkpoint==0.4.0

