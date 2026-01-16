
export CUDA_VISIBLE_DEVICES=0

MPS=${1:-0}

# Optional: start MPS for better multi-process sharing
if [[ $MPS -eq 1 ]] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[run] Starting CUDA MPS"
  sudo nvidia-cuda-mps-control -d || true
else
  echo "[run] Not starting CUDA MPS"
fi


nsys profile \
  -o two-process-ptr-crash-MPS_$MPS-$(date +%Y%m%d_%H%M%S) \
  --force-overwrite=true \
  --trace=cuda,nvtx,mpi,osrt \
  -s none --cpuctxsw=none \
  --gpuctxsw=true \
  --cuda-memory-usage=true \
  --cuda-um-gpu-page-faults=true \
  --cuda-um-cpu-page-faults=true \
  mpirun -np 2 ./two_process_ptr_crash


if [[ $MPS -eq 1 ]]; then
  echo quit | sudo nvidia-cuda-mps-control
fi
