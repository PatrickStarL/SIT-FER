"""Docker configuration for SIT-FER"""

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "scripts/train.py", "--config", "configs/base.yaml"]
