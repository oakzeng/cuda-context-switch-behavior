#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh [MiB] [stride_bytes] [iters] [use_managed_memory] [use_mps]
# Example: ./run.sh 4096 65536 5 1 1

MIB=${1:-16384}
STRIDE=${2:-65536}
ITERS=${3:-5}
USE_MANAGED_MEMORY=${4:-1}
MPS=${5:-0}

export CUDA_MODULE_LOADING=EAGER

# Optional: start MPS for better multi-process sharing
if [[ $MPS -eq 1 ]] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[run] Starting CUDA MPS"
  sudo nvidia-cuda-mps-control -d || true
else
  echo "[run] Not starting CUDA MPS"
fi


sudo nvidia-smi compute-policy --set-timeslice=1

nvidia-smi compute-policy -l

nsys profile \
  -o fault-observer-MPS_$MPS-$(date +%Y%m%d_%H%M%S) \
  --force-overwrite=true \
  --trace=cuda,nvtx,mpi,osrt \
  -s none --cpuctxsw=none \
  --gpuctxsw=true \
  --cuda-memory-usage=true \
  --cuda-um-gpu-page-faults=true \
  --cuda-um-cpu-page-faults=true \
  mpirun -quiet --bind-to none --tag-output \
    -np 1 -x CUDA_MODULE_LOADING=EAGER \
      ./faulter "$MIB" "$STRIDE" "$ITERS" "$USE_MANAGED_MEMORY" \
    : -np 1 -x CUDA_MODULE_LOADING=EAGER \
      ./observer 300 500


sudo nvidia-smi compute-policy --set-timeslice=0


nvidia-smi compute-policy -l

if [[ $MPS -eq 1 ]]; then
  echo quit | sudo nvidia-cuda-mps-control
fi

