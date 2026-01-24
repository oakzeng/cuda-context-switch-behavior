#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh [MiB] [stride_bytes] [iters]
# Example: ./run.sh 4096 65536 1

MIB=${1:-8192}
STRIDE=${2:-65536}
ITERS=${3:-2}
USE_MANAGED_MEMORY=${4:-0}
IN_KERNEL_ITERATION=${4:-25600000}

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


#mpirun -quiet --bind-to none --tag-output \
#  -np 1 -x CUDA_MODULE_LOADING=EAGER \
#    nsys profile \
#      -o fault_$(date +%Y%m%d_%H%M%S) \
#      --force-overwrite=true \
#      --trace=cuda,nvtx,mpi \
#      -s none --cpuctxsw=none \
#      --cuda-memory-usage=true \
#      --cuda-um-gpu-page-faults=true \
#      ./faulter "$MIB" "$STRIDE" "$ITERS" \
#  : -np 1 -x CUDA_MODULE_LOADING=EAGER \
#    nsys profile \
#      -o observer_$(date +%Y%m%d_%H%M%S) \
#      --force-overwrite=true \
#      --trace=cuda,nvtx,mpi \
#      -s none --cpuctxsw=none \
#      --cuda-memory-usage=true \
#      ./observer 300 1000
#


nsys profile \
  -o glmark2_cuda_managed_mem_$USE_MANAGED_MEMORY-$MIB-MiB-stride-$STRIDE-iter-$ITERS-inkerneliter-$IN_KERNEL_ITERATION-$(date +%Y%m%d_%H%M%S) \
  --force-overwrite=true \
  --trace=opengl,cuda,nvtx,mpi,osrt \
  -s none --cpuctxsw=none \
  --gpuctxsw=true \
  --cuda-memory-usage=true \
  --cuda-um-gpu-page-faults=true \
  --cuda-um-cpu-page-faults=true \
  --kill=none \
  mpirun --allow-run-as-root --bind-to none --oversubscribe --tag-output \
    --wdir "$PWD" \
    -np 1 -x CUDA_MODULE_LOADING=EAGER \
      ./faulter "$MIB" "$STRIDE" "$ITERS" "$USE_MANAGED_MEMORY" "$IN_KERNEL_ITERATION" \
    : -np 1 -x CUDA_MODULE_LOADING=EAGER \
    glmark2_mpi_wrapper 2

#mpirun --allow-run-as-root --bind-to none --oversubscribe --tag-output \
#  --wdir "$PWD" \
#  -np 1 -x CUDA_MODULE_LOADING=EAGER \
#  glmark2_mpi_wrapper


#nvidia-smi compute-policy --set-timeslice=0


#nvidia-smi compute-policy -l

