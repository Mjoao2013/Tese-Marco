#!/bin/bash
# Phase 4B MLGN Adaptive Lambda — HPC Deployment Script
# Usage: bash deploy_phase4b.sh
# Server: g08.hlt.inesc-id.pt | GPU: NVIDIA A100 80GB
# Ablation: fixes contrastive loss domination from Phase 4 via adaptive lambda scaling

set -e

REPO_DIR="/cfs/home/u037341/tese/Tese-Marco"
PHASE4B_DIR="$REPO_DIR/05_Phase4B"
SCRIPT="$PHASE4B_DIR/Scripts/train_phase4b_mlgn.py"
LOG="$PHASE4B_DIR/Results/phase4b_run.log"

echo "========================================"
echo "Phase 4B MLGN Adaptive Lambda — HPC"
echo "========================================"
echo "Repo:   $REPO_DIR"
echo "Script: $SCRIPT"
echo ""

cd "$REPO_DIR"
git fetch origin main
git checkout FETCH_HEAD -- 05_Phase4B/
echo "[git] 05_Phase4B/ updated from origin/main"

VENV_DIR="/cfs/home/u037341/tese/venv_phase4"
if [ -d "$VENV_DIR/bin" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[venv] Activated: $VENV_DIR"
else
    echo "ERROR: venv_phase4 not found at $VENV_DIR"
    echo "Expected the Phase 4 venv with PyTorch/CUDA already installed."
    exit 1
fi

python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo ""
echo "Starting Phase 4B MLGN training (adaptive lambda)..."

mkdir -p "$PHASE4B_DIR/Results"
mkdir -p "$PHASE4B_DIR/Models"

nohup python3 "$SCRIPT" > "$LOG" 2>&1 &
PID=$!
echo "Training PID: $PID"
echo "$PID" > "$PHASE4B_DIR/Results/train.pid"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG"
echo "  cat $PHASE4B_DIR/Results/phase4b_progress.txt"
echo ""
echo "Expected runtime: ~3-4 hours on A100"