#!/bin/bash
# Phase 4 MLGN — HPC Deployment Script
# Usage: bash deploy_phase4.sh
# Server: g08.hlt.inesc-id.pt | GPU: NVIDIA A100 80GB

set -e

REPO_DIR="/cfs/home/u037341/tese/Tese-Marco"
PHASE4_DIR="$REPO_DIR/05_Phase4"
SCRIPT="$PHASE4_DIR/Scripts/train_phase4_mlgn.py"
LOG="$PHASE4_DIR/Results/phase4_run.log"

echo "========================================"
echo "Phase 4 MLGN — Server Deployment"
echo "========================================"
echo "Repo:   $REPO_DIR"
echo "Script: $SCRIPT"
echo ""

cd "$REPO_DIR"
git pull origin main
echo "[git] Up to date"

VENV_DIR="$REPO_DIR/venv_phase3"
if [ -d "$VENV_DIR/bin" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[venv] Activated: $VENV_DIR"
else
    echo "[venv] Creating new venv..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip --quiet
    pip install -q torch transformers pandas scikit-learn numpy scipy skmultilearn
fi

python3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"

echo ""
echo "Starting Phase 4 MLGN training..."

mkdir -p "$PHASE4_DIR/Results"
mkdir -p "$PHASE4_DIR/Models"

nohup python3 "$SCRIPT" > "$LOG" 2>&1 &
PID=$!
echo "Training PID: $PID"
echo "$PID" > "$PHASE4_DIR/Results/train.pid"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG"
echo "  cat $PHASE4_DIR/Results/phase4_progress.txt"
echo ""
echo "Expected runtime: ~3-4 hours on A100"
