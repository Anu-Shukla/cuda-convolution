#include <cuda_runtime.h>

constexpr int MAX_FILTER_ELEMENTS = 225;

__constant__ float c_filter[MAX_FILTER_ELEMENTS];

struct Conv2DConfig {
    int height;
    int width;
    int filter_size;
    int padding;
    int stride;
};

__global__ void conv2d_constant_kernel(const float* input,
                                       float* output,
                                       Conv2DConfig cfg) {
    // TODO: One thread computes one output element.
    // TODO: Use the same indexing pattern as the naive kernel.
    // TODO: Read filter values from c_filter instead of global memory.
    // TODO: Write the accumulated value to output.
}

void launch_constant_conv2d(const float* d_input,
                            const float* h_filter,
                            float* d_output,
                            Conv2DConfig cfg) {
    // TODO: Copy h_filter into c_filter with cudaMemcpyToSymbol.
    // TODO: Compute output dimensions from cfg.
    // TODO: Choose block and grid dimensions.
    // TODO: Launch conv2d_constant_kernel.
}

int main(int argc, char** argv) {
    // TODO: Parse shape arguments.
    // TODO: Allocate and initialize host/device memory.
    // TODO: Launch constant-memory convolution.
    // TODO: Copy results back and validate.
    // TODO: Add timing when the kernel is implemented.
    return 0;
}
