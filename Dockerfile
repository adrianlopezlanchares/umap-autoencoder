FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# ---- System + Python 3.11 ----
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git wget curl ffmpeg build-essential libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# ---- Install PyTorch CUDA 12.1 compatible ----
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ---- Copy and install Python dependencies ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app/

ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

CMD ["bash"]