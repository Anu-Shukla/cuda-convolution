#include <cuda_runtime.h>
#include <stdio.h>

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
    extern __shared__ float tile[];
    //shared tile dimensions:
    int tile_h = (blockDim.y - 1) * cfg.stride + cfg.filter_size;
    int tile_w = (blockDim.x - 1) * cfg.stride + cfg.filter_size;

    //where the tile needs to be loaded from:
    int out_tile_row = blockIdx.y * blockDim.y;
    int out_tile_col = blockIdx.x * blockDim.x;

    int input_tile_start_row = out_tile_row * cfg.stride - cfg.padding;
    int input_tile_start_col = out_tile_col * cfg.stride - cfg.padding;

    //cooperative load tile:
    for (int tile_row = threadIdx.y; tile_row < tile_h; tile_row += blockDim.y) {
        for (int tile_col = threadIdx.x; tile_col < tile_w; tile_col += blockDim.x) { 
            int in_row = input_tile_start_row + tile_row;
            int in_col = input_tile_start_col + tile_col;

            int tile_index = tile_row * tile_w + tile_col;

            if (in_row >= 0 && in_row < cfg.height &&
                    in_col >= 0 && in_col < cfg.width) {
                int input_index = in_row * cfg.width + in_col;
                tile[tile_index] = input[input_index];
            } else {
                tile[tile_index] = 0.0f;
            }
        }
    }
    __syncthreads();

    //threads output:
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    int out_h = (cfg.height - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;
    int out_w = (cfg.width - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;

    float accumulator = 0.0f;
    if (out_row < out_h && out_col < out_w) {
        int tile_filter_start_row = threadIdx.y * cfg.stride;
        int tile_filter_start_col = threadIdx.x * cfg.stride;

        for (int filter_row = 0; filter_row < cfg.filter_size; filter_row++) {
            for (int filter_col = 0; filter_col < cfg.filter_size; filter_col++) {
                int tile_read_row = tile_filter_start_row + filter_row;
                int tile_read_col = tile_filter_start_col + filter_col;

                int tile_read_index = tile_read_row * tile_w + tile_read_col;
                int filter_index = filter_row * cfg.filter_size + filter_col;
                accumulator += tile[tile_read_index] * filter[filter_index];
            }
        }
        int output_index = out_row * out_w + out_col;
        output[output_index] = accumulator;
    }
}

void launch_tiled_conv2d(const float* d_input,
                         const float* d_filter,
                         float* d_output,
                         Conv2DConfig cfg) { 
    int out_h = (cfg.height - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;
    int out_w = (cfg.width - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;

    dim3 block(16,16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

    int tile_h = (block.y - 1) * cfg.stride + cfg.filter_size;
    int tile_w = (block.x - 1) * cfg.stride + cfg.filter_size;
    int shared_bytes = tile_h * tile_w * sizeof(float);

    conv2d_tiled_kernel<<<grid, block, shared_bytes>>>(d_input, d_filter, d_output, cfg);
}

int main(int argc, char** argv) {
    //NOTE: Hardcoded values for cfg, input matrix, and filters:
    Conv2DConfig cfg;
    cfg.height = 2048;
    cfg.width = 2048;
    cfg.filter_size = 21;
    cfg.padding = cfg.filter_size / 2;
    cfg.stride = 1;

    int out_h = (cfg.height - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;
    int out_w = (cfg.width - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;

    int input_elems = cfg.height * cfg.width;
    int filter_elems = cfg.filter_size * cfg.filter_size;
    int output_elems = out_h * out_w;

    float* h_input = new float[input_elems];
    float* h_filter = new float[filter_elems];
    float* h_output = new float[output_elems];

    for (int i = 0; i < input_elems; i++) {
        h_input[i] = 1.0f;
    }

    for (int i = 0; i < filter_elems; i++) {
        h_filter[i] = 1.0f;
    }

    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, input_elems * sizeof(float));
    cudaMalloc(&d_filter, filter_elems * sizeof(float));
    cudaMalloc(&d_output, output_elems * sizeof(float));

    cudaMemcpy(d_input, h_input, input_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_elems * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    //warmup run:
    launch_tiled_conv2d(d_input, d_filter, d_output, cfg);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    launch_tiled_conv2d(d_input, d_filter, d_output, cfg);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop); 

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
          printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time: %.3f ms\n", ms);
    double gflops = (2.0 * output_elems * filter_elems) / (ms * 1.0e6);
    printf("GFLOP/s: %.2f\n", gflops);

    cudaMemcpy(h_output, d_output, output_elems * sizeof(float), cudaMemcpyDeviceToHost);

    /**
    for (int r = 0; r < out_h; ++r) {
          for (int c = 0; c < out_w; ++c) {
              printf("%6.1f ", h_output[r * out_w + c]);
          }
          printf("\n");
    }**/

    delete[] h_input; delete[] h_filter; delete[] h_output;
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
