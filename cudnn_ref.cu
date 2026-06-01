#include <cuda_runtime.h>
#include <cudnn.h>

struct Conv2DConfig {
    int height;
    int width;
    int filter_size;
    int padding;
    int stride;
};

void run_cudnn_conv2d(const float* d_input,
                      const float* d_filter,
                      float* d_output,
                      Conv2DConfig cfg) {
    // TODO: Create cuDNN handle.
    // TODO: Create input, filter, output, and convolution descriptors.
    // TODO: Configure descriptors for NCHW with N=1 and C=1.
    // TODO: Select a forward convolution algorithm.
    // TODO: Allocate workspace if needed.
    // TODO: Call cudnnConvolutionForward.
    // TODO: Destroy descriptors, workspace, and handle.
}

int main(int argc, char** argv) {
    // TODO: Parse shape arguments.
    // TODO: Allocate and initialize host/device memory.
    // TODO: Run cuDNN convolution.
    // TODO: Copy results back and compare against your CUDA kernels.
    // TODO: Add timing and GFLOP/s reporting.
    return 0;
}
