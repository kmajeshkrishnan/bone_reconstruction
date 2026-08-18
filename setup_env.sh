#!/bin/bash

# Set up environment for Bone Reconstruction project
# Usage: source setup_env.sh

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add extension directories to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/extensions/chamfer_dist:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}/extensions/expansion_penalty:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}/extensions/KNN_CUDA:${PYTHONPATH}"
export PYTHONPATH="${SCRIPT_DIR}/extensions/Pointnet2/pointnet2:${PYTHONPATH}"

# Add PyTorch libraries to LD_LIBRARY_PATH
CONDA_ENV_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="${CONDA_ENV_PATH}:${LD_LIBRARY_PATH}"

echo "Environment variables set successfully!"
echo "PYTHONPATH includes extensions"
echo "LD_LIBRARY_PATH includes PyTorch libraries"
