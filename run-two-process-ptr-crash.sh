
export CUDA_VISIBLE_DEVICES=0

nsys profile \
  -o two-process-ptr-crash-$(date +%Y%m%d_%H%M%S) \
  --force-overwrite=true \
  --trace=cuda,nvtx,mpi,osrt \
  -s none --cpuctxsw=none \
  --gpuctxsw=true \
  --cuda-memory-usage=true \
  --cuda-um-gpu-page-faults=true \
  --cuda-um-cpu-page-faults=true \
  mpirun -np 2 ./two_process_ptr_crash
