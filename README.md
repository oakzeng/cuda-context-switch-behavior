# Proving GPU Context Switch During Page Fault on Linux (NVIDIA Pascal+)

This experiment demonstrates that **while one CUDA context is stalled by GPU page faults**, the GPU can **context-switch and run work from another context**. It uses **Unified Memory (UVM)** to deliberately trigger page faults from the GPU, and a second process repeatedly launches short kernels and measures their latency.

> **Background**
>
> - Starting with **Pascal (GP100)**, NVIDIA GPUs support **fine-grained compute and graphics preemption**, allowing the driver to **save/restore execution state and switch contexts while a kernel is in-flight**. See the Pascal whitepaper (section on *Unified Memory and Compute Preemption*).
> - NVIDIA’s **context switching subsystem** saves global and per-GPC state into context images, enabling rapid switching across contexts and workloads, with multiple preemption granularities (instruction-level, CTA-level, WFI). See NVIDIA/open-gpu-doc.
> - In **Unified Virtual Memory**, a GPU page fault traps to the driver/UVM. The faulting warp stalls while data is migrated/mapped; other contexts can proceed, which is what this test shows. See NVIDIA open-gpu-kernel-modules discussion of page-fault workflow.
>
> **References:**
> - NVIDIA Pascal Architecture Whitepaper: https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf
> - Context Switching internals (TU104): https://deepwiki.com/NVIDIA/open-gpu-doc/4.2-context-switching
> - Page-fault workflow (UVM): https://github.com/NVIDIA/open-gpu-kernel-modules/discussions/619
> - Blackwell tuning/compatibility (preemption model retained): https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html

## What the programs do

- `faulter.cu` allocates a large **managed** buffer, sets its **preferred location to CPU**, prefetches it to CPU, then launches a kernel that touches the buffer at **64 KiB strides**. Each first touch on GPU triggers a **page fault and migration**, stalling the faulting warp repeatedly.
- `observer.cu` in a separate process uses a **high-priority stream** to launch a tiny kernel many times and **records per-launch latency**. If **context switching during the fault** is happening, you will see **observer kernels completing steadily** while `faulter` is still running.

## Build

Requires CUDA toolkit on Linux.

```bash
make
```

## Run

Run both processes via the helper script:

```bash
./run.sh 4096 65536 1   # 4 GiB, 64 KiB stride, 1 iteration
```

This will:
1. Optionally start **CUDA MPS** (if available) to improve multi-process sharing (not strictly required).
2. Launch `faulter` in the background to induce GPU page faults.
3. Launch `observer` to measure short-kernel latency while faults are ongoing.
4. Write logs: `faulter.log` and `observer_times.csv`.

## Expected evidence

- `faulter.log` will show a long-running kernel (hundreds/thousands of ms depending on memory size/stride).
- `observer_times.csv` will show **sub-ms to few-ms** per-iteration times **continuing to complete** while `faulter` is in-flight. This demonstrates the scheduler **context-switches away from the faulting context** to run `observer`.

For a visual timeline, capture with **Nsight Systems**:

```bash
nsys profile -o ctxswitch --stats=true ./observer 500 5000 &
nsys profile -o fault --stats=true ./faulter 4096 65536 1
```

You should see overlapping GPU activity with `observer` kernels interleaving while `faulter` experiences migration stalls.

## Tips & Variations

- Increase `MIB` (buffer size) or reduce `stride` to create **more faults** and longer stalls.
- Try with and without **MPS**: `nvidia-cuda-mps-control -d` to start, `echo quit | nvidia-cuda-mps-control` to stop.
- On systems with **MIG** (Ampere+), placing each process in a different MIG slice isolates them (less contention) and may reduce visibility of preemption.
- Use `CUDA_VISIBLE_DEVICES` to pin both programs to the same GPU.

## Why this proves it

- The faulting kernel’s warp stalls on each first-touch; servicing a UVM fault takes the driver and copy engine time.
- While stalled, the GPU hardware and driver can **save the context state** and **schedule another context** (observer). The presence of continued, steady `observer` completions while `faulter` is active is practical evidence of **context switching during page fault handling**.

## Caveats

- Exact fault granularity depends on driver/GPU (UVM can operate on 64 KiB–2 MiB blocks).
- If the GPU is fully saturated by non-faulting compute (e.g., massive occupancy), observer latency will rise; tune stride and memory size to ensure fault-induced stalls.
- Some older drivers/devices may coalesce migrations, reducing visible stalls; adjust parameters accordingly.

