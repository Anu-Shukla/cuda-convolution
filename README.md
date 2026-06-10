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

| Implementation | Time (ms) |
| --- | ---: |
| Naive CUDA | 1.527 |
| Tiled CUDA | 1.537 |
| Constant Memory CUDA | 1.179 |
| cuDNN Reference | 1.791 |
