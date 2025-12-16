#!/bin/bash

# Server Deployment Setup Script

# 1. Create Conda Environment
echo "Creating Conda environment 'fire_damage_cls'..."
conda env create -f environment.yml

# 2. Activate Environment
echo "To activate the environment, run:"
echo "conda activate fire_damage_cls"

# 3. Verify GPU Availability
echo "Verifying GPU setup..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fire_damage_cls
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"

echo "Setup Complete."
