# GRM Dockerfile
# Usage:
#   docker build -f docker/Dockerfile -t grm:latest .
#   docker run --gpus all -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/logs:/app/logs grm:latest

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    build-essential \
    binutils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install additional dependencies
RUN pip3 install \
    tqdm \
    numpy \
    pandas \
    torchvision

# Set working directory
WORKDIR /app

# Copy project files
COPY grm ./grm
COPY examples ./examples
COPY datasets ./datasets
COPY data ./data
COPY setup.py .
COPY hardware_presets.json .
COPY *.md .

# Build CUDA extension
RUN python3 setup.py build_ext --inplace

# Verify CUDA extension
RUN python3 -c "from grm.core.cuda_ops import get_grm_cuda_runtime_status; print(get_grm_cuda_runtime_status())"

# Create directories for checkpoints and logs
RUN mkdir -p checkpoints logs

# Default command
CMD ["python3", "grm/utils/train.py", "--help"]
