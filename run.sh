#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh [MiB] [stride_bytes] [iters]
# Example: ./run.sh 4096 65536 1

MIB=${1:-16384}
STRIDE=${2:-65536}
ITERS=${3:-5}
USE_MANAGED_MEMORY=${4:-1}

export CUDA_MODULE_LOADING=EAGER

# Optional: start MPS for better multi-process sharing
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[run] Not Starting CUDA MPS"
  #nvidia-cuda-mps-control -d || true
else
  echo "[run] CUDA MPS not found; continuing without it"
fi


#nvidia-smi compute-policy --set-timeslice=1

nvidia-smi compute-policy -l
#CUDA_MODULE_LOADING=EAGER nsys profile -o fault_$(date +%Y%m%d_%H%M%S) --force-overwrite=true --trace=cuda,nvtx -s none --cpuctxsw=none --cuda-memory-usage=true --cuda-um-gpu-page-faults=true ./faulter "$MIB" "$STRIDE" "$ITERS" &
#CUDA_MODULE_LOADING=EAGER nsys profile -o observer_$(date +%Y%m%d_%H%M%S) --force-overwrite=true --trace=cuda,nvtx -s none --cpuctxsw=none --cuda-memory-usage=true ./observer 1000000 500 | tee -a  observer_times.csv

#mpirun -np 1 ./faulter "$MIB" "$STRIDE" "$ITERS" : -np 1 ./observer 1000 500
#./faulter "$MIB" "$STRIDE" "$ITERS" &
#./observer 1000000 500 | tee -a  observer_times.csv


mpirun -quiet --bind-to none --tag-output \
  -np 1 -x CUDA_MODULE_LOADING=EAGER \
    nsys profile \
      -o fault_$(date +%Y%m%d_%H%M%S) \
      --force-overwrite=true \
      --trace=cuda,nvtx,mpi \
      -s none --cpuctxsw=none \
      --cuda-memory-usage=true \
      --cuda-um-gpu-page-faults=true \
      ./faulter "$MIB" "$STRIDE" "$ITERS" \
  : -np 1 -x CUDA_MODULE_LOADING=EAGER \
    nsys profile \
      -o observer_$(date +%Y%m%d_%H%M%S) \
      --force-overwrite=true \
      --trace=cuda,nvtx,mpi \
      -s none --cpuctxsw=none \
      --cuda-memory-usage=true \
      ./observer 300 1000




#nvidia-smi compute-policy --set-timeslice=0


#nvidia-smi compute-policy -l

