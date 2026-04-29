#!/usr/bin/env bash
# ===========================================================================
# AWS EC2 Setup Script for DDI Study
# ===========================================================================
# Run this on a fresh EC2 instance (Ubuntu 22.04 or Deep Learning AMI).
#
# Usage:
#   1. Launch an EC2 instance (recommended: r6i.2xlarge for 64GB RAM)
#   2. SSH in:  ssh -i your-key.pem ubuntu@<public-ip>
#   3. Run:  cd WI-SP26-DSC-Capstone && chmod +x aws_setup.sh && ./aws_setup.sh
#
# After setup completes, start a tmux session and run:
#   tmux new -s ddi
#   cd ~/WI-SP26-DSC-Capstone
#   source .venv/bin/activate
#   python download_faers.py
#   python ddi_study.py
# ===========================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "============================================"
echo "  DDI Study - EC2 Environment Setup"
echo "============================================"

# ------------------------------------------
# 1. System packages
# ------------------------------------------
echo ""
echo "[1/6] Installing system packages ..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    git tmux htop unzip curl wget \
    build-essential libxrender1 libxext6

# ------------------------------------------
# 2. Python virtual environment
# ------------------------------------------
echo ""
echo "[2/6] Setting up Python virtual environment ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# ------------------------------------------
# 3. Install Python dependencies
# ------------------------------------------
echo ""
echo "[3/6] Installing Python packages ..."

pip install \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    requests \
    rapidfuzz

pip install rdkit

# PyTorch - install GPU version if CUDA is available, CPU otherwise
if command -v nvidia-smi &> /dev/null; then
    echo "  CUDA detected - installing PyTorch with GPU support ..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    echo "  No CUDA - installing CPU-only PyTorch ..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# ------------------------------------------
# 4. Create output directories
# ------------------------------------------
echo ""
echo "[4/6] Creating output directories ..."
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/results/ddi_study"
mkdir -p "$PROJECT_DIR/reports"

# ------------------------------------------
# 5. Verify installation
# ------------------------------------------
echo ""
echo "[5/6] Verifying installation ..."
python3 << 'PYEOF'
import pandas, numpy, scipy, sklearn, matplotlib, requests, rapidfuzz
print(f"  pandas     {pandas.__version__}")
print(f"  numpy      {numpy.__version__}")
print(f"  scipy      {scipy.__version__}")
print(f"  sklearn    {sklearn.__version__}")
print(f"  matplotlib {matplotlib.__version__}")
print(f"  rapidfuzz  {rapidfuzz.__version__}")
PYEOF

python3 << 'PYEOF'
from rdkit import Chem
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
print(f"  rdkit      OK (aspirin parsed: {mol is not None})")
PYEOF

python3 << 'PYEOF'
import torch
print(f"  torch      {torch.__version__}")
print(f"  CUDA       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        {torch.cuda.get_device_name(0)}")
PYEOF

# ------------------------------------------
# 6. Summary
# ------------------------------------------
echo ""
echo "[6/6] Done!"
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Project:  $PROJECT_DIR"
echo "  Venv:     source $VENV_DIR/bin/activate"
echo ""
echo "  Next steps:"
echo "    tmux new -s ddi"
echo "    cd $PROJECT_DIR"
echo "    source .venv/bin/activate"
echo ""
echo "    # Download full FAERS (all years):"
echo "    python download_faers.py"
echo ""
echo "    # Or download a specific range:"
echo "    python download_faers.py --years 2020 2021 2022 2023 2024 2025"
echo ""
echo "    # Resume if interrupted:"
echo "    python download_faers.py --resume"
echo ""
echo "    # Run the DDI study:"
echo "    python ddi_study.py"
echo ""
echo "  Tip: Use tmux so the job survives SSH disconnects."
echo "    Detach: Ctrl+B then D"
echo "    Reattach: tmux attach -t ddi"
echo "============================================"
