
# VADAR: Visual Agentic AI for Spatial Reasoning with a Dynamic API

## Research Paper Implementation
This repository contains the official implementation of the CVPR 2025 research paper  
**“Visual Agentic AI for Spatial Reasoning with a Dynamic API”**.

## Overview
VADAR is a visual agentic AI framework for complex spatial and 3D reasoning from images.  
It dynamically generates APIs and executable programs using an agent-based pipeline that combines vision models and large language models to answer visual questions.

## How to Run

```bash
# Clone the repository
git clone https://github.com/damianomarsili/VADAR.git
cd VADAR

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies and download models
sh setup.sh

# Add your OpenAI API key
echo YOUR_OPENAI_API_KEY > api.key

# Download evaluation datasets
sh download_data.sh

# Run evaluation (default: Omni3D-Bench)
python evaluate.py \
  --annotations-json data/[DATASET_NAME]/annotations.json \
  --image-pth data/[DATASET_NAME]/images/

# Optional dataset flags
# --dataset clevr
# --dataset gqa
