#!/bin/bash
# ISCTE Phase 2 ModernBERT - Complete Deployment Script
# Usage: bash deploy_phase2.sh <path_to_dataset>

set -e

DATASET_PATH=${1:-.}
WORK_DIR=~/tese_modernbert_p2
mkdir -p $WORK_DIR/{data,models,logs,scripts}
cd $WORK_DIR

echo "========================================"
echo "ModernBERT Phase 2 - Server Setup"
echo "========================================"
echo "Working directory: $WORK_DIR"
echo "Dataset path: $DATASET_PATH"
echo ""

# Activate environment
if [ -d "venv_p2/bin" ]; then
    source venv_p2/bin/activate
else
    python3 -m venv venv_p2
    source venv_p2/bin/activate
fi

# Upgrade pip
pip install --upgrade pip --quiet

# Install dependencies silently
echo "[1/4] Installing PyTorch + dependencies..."
pip install -q torch transformers pandas scikit-learn numpy matplotlib seaborn tensorboard tqdm --no-cache-dir

echo "[2/4] Testing imports..."
python3 -c "import torch, transformers; print(f'✓ PyTorch {torch.__version__}'); print(f'✓ CUDA: {torch.cuda.is_available()}')"

echo "[3/4] Dataset check..."
if [ -f "$DATASET_PATH/bgg_geektype_subset.parquet" ]; then
    echo "✓ Dataset found"
    cp "$DATASET_PATH/bgg_geektype_subset.parquet" data/
else
    echo "⚠ Dataset not found at $DATASET_PATH"
    echo "  Place dataset at: $WORK_DIR/data/bgg_geektype_subset.parquet"
    exit 1
fi

echo "[4/4] Ready for training!"
echo ""
echo "Start training:"
echo "  cd $WORK_DIR"
echo "  source venv_p2/bin/activate"
echo "  python scripts/train_modernbert.py"
