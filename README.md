# CUDA 2D Convolution

Goal: implement 2D convolution in CUDA across several optimization levels, then
compare against cuDNN.

## Files

- `naive.cu`: one thread per output element, all global memory
- `tiled.cu`: input tiles loaded into shared memory
- `constant.cu`: filter stored in CUDA constant memory
- `cudnn_ref.cu`: cuDNN `cudnnConvolutionForward` reference path

## Formula

```text
output_size = (N - F + 2P) / S + 1
```

where:

- `N`: input size
- `F`: filter size
- `P`: padding
- `S`: stride

## Results

Configuration: `2048x2048` input, `21x21` filter, padding `10`, stride `1`, FP32.

| Implementation | Time (ms) | GFLOP/s |
| --- | ---: | ---: |
| Naive CUDA | 1.538 | 2405.04 |
| Tiled CUDA | 1.536 | 2407.90 |
| Constant Memory CUDA | 1.177 | 3144.19 |
| cuDNN Reference | 1.795 | 2061.15 |

## Profiling

Profiled the fastest implementation, `constant`, with Nsight Compute:

| Metric | Value |
| --- | ---: |
| Kernel duration | 1.42 ms |
| Compute throughput | 81.98% |
| Memory throughput | 64.16% |
| DRAM throughput | 2.21% |
| L1/TEX hit rate | 98.80% |
| L2 hit rate | 86.94% |
| Achieved occupancy | 87.77% |
| Registers per thread | 28 |

The profile indicates that the constant-memory kernel is mostly compute/instruction limited rather than DRAM-bandwidth limited. The low DRAM throughput and high cache hit rates show that most memory accesses are being served efficiently from cache.
