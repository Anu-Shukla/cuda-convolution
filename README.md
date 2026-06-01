# Project 3: CUDA 2D Convolution

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

## Build Commands

```bash
nvcc naive.cu -o naive -O3 --use_fast_math
nvcc tiled.cu -o tiled -O3 --use_fast_math
nvcc constant.cu -o constant -O3 --use_fast_math
nvcc cudnn_ref.cu -o cudnn_ref -O3 -lcudnn
```

## Suggested Order

1. Fill in `naive.cu`.
2. Fill in `tiled.cu`.
3. Fill in `constant.cu`.
4. Fill in `cudnn_ref.cu`.
