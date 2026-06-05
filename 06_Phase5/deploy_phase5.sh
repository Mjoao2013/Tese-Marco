#!/bin/bash
# Phase 5: Mechanism Classification — HPC Deployment Script
# Usage: bash deploy_phase5.sh [5a|5b|5c]
# Default: runs Phase 5A (BERT-BR baseline) if no argument given.
#
# Scripts:
#   5a — train_phase5a_bert_br.py  : BERT-BR baseline (same config as Phase 3 Enhanced)
#   5b — train_phase5b_focal.py    : BERT-BR + Focal Loss (γ=2) ablation
#   5c — train_phase5c_mlgn.py     : MLGN adaptive λ (same as Phase 4B, mechanisms data)
#
# Recommended run order: 5a first, then 5b, then 5c.
# Each takes ~3–5 hours on A100. Do not queue all three simultaneously
# unless you have confirmed 5a completes without error.
#
# Server: g08.hlt.inesc-id.pt | GPU: NVIDIA A100 80GB PCIe

set -e

PHASE="${1:-5a}"   # default to 5a if no argument provided

REPO_DIR="/cfs/home/u037341/tese/Tese-Marco"
PHASE5_DIR="$REPO_DIR/06_Phase5"

case "$PHASE" in
    5a)
        SCRIPT="$PHASE5_DIR/Scripts/train_phase5a_bert_br.py"
        LOG="$PHASE5_DIR/Results/phase5a_run.log"
        PID_FILE="$PHASE5_DIR/Results/phase5a_train.pid"
        DESC="Phase 5A — BERT-BR Baseline"
        ;;
    5b)
        SCRIPT="$PHASE5_DIR/Scripts/train_phase5b_focal.py"
        LOG="$PHASE5_DIR/Results/phase5b_run.log"
        PID_FILE="$PHASE5_DIR/Results/phase5b_train.pid"
        DESC="Phase 5B — Focal Loss Ablation"
        ;;
    5c)
        SCRIPT="$PHASE5_DIR/Scripts/train_phase5c_mlgn.py"
        LOG="$PHASE5_DIR/Results/phase5c_run.log"
        PID_FILE="$PHASE5_DIR/Results/phase5c_train.pid"
        DESC="Phase 5C — MLGN Adaptive Lambda"
        ;;
    *)
        echo "ERROR: Unknown phase '$PHASE'. Use: 5a | 5b | 5c"
        exit 1
        ;;
esac

echo "========================================"
echo "$DESC — HPC Deployment"
echo "========================================"
echo "Repo:   $REPO_DIR"
echo "Script: $SCRIPT"
echo "Log:    $LOG"
echo ""

# Sync latest code from git
cd "$REPO_DIR"
git fetch origin main
git checkout FETCH_HEAD -- 06_Phase5/
echo "[git] 06_Phase5/ updated from origin/main"

# Activate venv (reuse Phase 4 venv — same dependencies)
VENV_DIR="/cfs/home/u037341/tese/venv_phase4"
if [ -d "$VENV_DIR/bin" ]; then
    source "$VENV_DIR/bin/activate"
    echo "[venv] Activated: $VENV_DIR"
else
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Expected the Phase 4 venv with PyTorch/CUDA, transformers, skmultilearn."
    exit 1
fi

# Sanity-check environment
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
import skmultilearn; print('skmultilearn: OK')
import transformers; print(f'transformers: {transformers.__version__}')
"

# Verify data file exists before launching
python3 -c "
import pandas as pd
from pathlib import Path
p = Path('/cfs/home/u037341/tese/Tese-Marco/Dataset/XML Dataset/bgg_mechanism_subset.parquet')
assert p.exists(), f'Data file not found: {p}'
df = pd.read_parquet(p)
print(f'Data OK: {df.shape[0]:,} games, columns: {list(df.columns)}')
assert 'mechanisms_list' in df.columns, 'mechanisms_list column missing'
print('mechanisms_list column: OK')
"

echo ""
echo "Starting $DESC..."
mkdir -p "$PHASE5_DIR/Results"
mkdir -p "$PHASE5_DIR/Models"

nohup python3 "$SCRIPT" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
echo "Training PID: $PID  (saved to $PID_FILE)"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG"
echo "  cat $PHASE5_DIR/Results/phase${PHASE}_progress.txt"
echo ""
echo "Expected runtime: ~3–5 hours on A100"
echo ""
echo "When complete, pull results:"
echo "  git -C $REPO_DIR add 06_Phase5/Results/ 06_Phase5/Models/"
echo "  git -C $REPO_DIR commit -m 'Phase ${PHASE} results'"
