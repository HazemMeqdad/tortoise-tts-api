FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and setup files
COPY requirements_api.txt requirements.txt setup.py ./

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.9.1)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies first to avoid conflicts
RUN pip3 install --no-cache-dir transformers==4.31.0 tokenizers==0.13.3

# Install API requirements
RUN pip3 install --no-cache-dir -r requirements_api.txt

# Copy the entire project
COPY . .

# Install the project with no dependencies (since we installed them manually)
RUN pip3 install --no-cache-dir --no-deps -e .

# Expose port for API
EXPOSE 8000

# Set default command to run the API
CMD ["python3", "api.py"]
