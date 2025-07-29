# Gitpod Dockerfile for IoT Edge Graph Anomaly Detection development
FROM gitpod/workspace-python-3.11

# Install system dependencies for ML and IoT development
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libpng-dev \
    libswscale-dev \
    libtiff-dev \
    pkg-config \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Docker for container development
RUN curl -fsSL https://get.docker.com | sh \
    && usermod -aG docker gitpod

# Install additional development tools
RUN pip install --upgrade pip setuptools wheel

USER gitpod

# Pre-install common ML dependencies to speed up workspace startup
RUN pip install --user \
    torch \
    torch-geometric \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov

# Configure Git for better development experience
RUN git config --global pull.rebase false \
    && git config --global init.defaultBranch main

# Set up shell environment
RUN echo 'alias ll="ls -la"' >> ~/.bashrc \
    && echo 'alias python="python3"' >> ~/.bashrc \
    && echo 'alias pip="pip3"' >> ~/.bashrc