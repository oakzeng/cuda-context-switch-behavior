
#!/usr/bin/env bash
# Minimal runner for the CUDA+MPI MPS demo.
# Usage:
#   ./run_simple.sh one-no-mps
#   ./run_simple.sh two-no-mps
#   ./run_simple.sh two-mps
#   ./run_simple.sh all 
#
# Optional quick tweaks (edit below):
BIN="./mps"
GPU=0
RANKS=2
MPIRUN="mpirun"

set -e

PLAN="${1:-}"
if [[ -z "$PLAN" ]]; then
  echo "Usage: $0 {one-no-mps|two-no-mps|two-mps|all}"
  exit 1
fi

# Keep all ranks on the same GPU.
export CUDA_VISIBLE_DEVICES="$GPU"

one_no_mps() {
  # (Very simple) try to ensure MPS isn't running; ignore errors if it isn't.
  echo quit | sudo nvidia-cuda-mps-control >/dev/null 2>&1 || true
  sudo nvidia-smi -i "$GPU" -c DEFAULT >/dev/null 2>&1 || true

  $MPIRUN -n 1 ./mps
}

two_no_mps() {
  echo quit | sudo nvidia-cuda-mps-control >/dev/null 2>&1 || true
  sudo nvidia-smi -i "$GPU" -c DEFAULT >/dev/null 2>&1 || true

nsys profile \
  -o two-rank-MPS_0-$(date +%Y%m%d_%H%M%S) \
  --force-overwrite=true \
  --trace=cuda,nvtx,mpi,osrt \
  -s none --cpuctxsw=none \
  --gpuctxsw=true \
  --cuda-memory-usage=true \
  --cuda-um-gpu-page-faults=true \
  --cuda-um-cpu-page-faults=true \
  $MPIRUN -n "$RANKS" ./mps
}

two_mps() {
  # Enable MPS (requires sudo). Keep it as short and simple as possible.
  sudo nvidia-smi -i "$GPU" -c EXCLUSIVE_PROCESS
  sudo nvidia-cuda-mps-control -d
  sleep 1

nsys profile \
  -o two-rank-MPS_1-$(date +%Y%m%d_%H%M%S) \
  --force-overwrite=true \
  --trace=cuda,nvtx,mpi,osrt \
  -s none --cpuctxsw=none \
  --gpuctxsw=true \
  --cuda-memory-usage=true \
  --cuda-um-gpu-page-faults=true \
  --cuda-um-cpu-page-faults=true \
  $MPIRUN -n "$RANKS" ./mps

  # Tear down MPS and restore default mode.
  echo quit | sudo nvidia-cuda-mps-control
  sudo nvidia-smi -i "$GPU" -c DEFAULT
}


case "$PLAN" in
  one-no-mps) one_no_mps ;;
  two-no-mps) two_no_mps ;;
  two-mps)    two_mps ;;
  all)    one_no_mps ; two_no_mps ; two_mps ;;
  *) echo "Unknown plan: $PLAN"; exit 1 ;;
esac

