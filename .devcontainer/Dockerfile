FROM almalinux:9

# Install basic dependencies
RUN dnf install -y --allowerasing \
    git \
    curl \
    wget \
    which \
    tar \
    gzip \
    bzip2 \
    ca-certificates \
    && dnf clean all

# Install OpenCode
RUN curl -fsSL https://opencode.ai/install | bash

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh

# Add Pixi to PATH
ENV PATH="/root/.pixi/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Verify installations
RUN pixi --version || echo "Pixi installation will be verified on container start"
