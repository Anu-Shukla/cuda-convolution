#include <cuda_runtime.h>

struct Conv2DConfig {
    int height;
    int width;
    int filter_size;
    int padding;
    int stride;
};

__global__ void conv2d_tiled_kernel(const float* input,
                                    const float* filter,
                                    float* output,
                                    Conv2DConfig cfg) {
    // TODO: Use extern __shared__ memory for the input tile.
    // TODO: Compute this block's output tile origin.
    // TODO: Cooperatively load the required input region into shared memory.
    // TODO: Synchronize threads after loading the tile.
    // TODO: Compute one output element per thread from shared memory.
    // TODO: Handle padding, stride, and boundary checks.
}

void launch_tiled_conv2d(const float* d_input,
                         const float* d_filter,
                         float* d_output,
                         Conv2DConfig cfg) {
    // TODO: Compute output dimensions from cfg.
    // TODO: Choose tile, block, grid, and shared-memory size.
    // TODO: Launch conv2d_tiled_kernel.
}

int main(int argc, char** argv) {
    // TODO: Parse shape arguments.
    // TODO: Allocate and initialize host/device memory.
    // TODO: Launch tiled convolution.
    // TODO: Copy results back and validate.
    // TODO: Add timing when the kernel is implemented.
    return 0;
}
