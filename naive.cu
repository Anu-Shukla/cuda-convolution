#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

struct Conv2DConfig {
    int height;
    int width;
    int filter_size;
    int padding;
    int stride;
};

void conv2d_cpu_reference(const float* input,
                          const float* filter,
                          float* output,
                          Conv2DConfig cfg) {
    int out_h = (cfg.height - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;
    int out_w = (cfg.width - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;

    for (int out_row = 0; out_row < out_h; out_row++) {
        for (int out_col = 0; out_col < out_w; out_col++) {
            float accumulator = 0.0f;

            for (int filter_row = 0; filter_row < cfg.filter_size; filter_row++) {
                for (int filter_col = 0; filter_col < cfg.filter_size; filter_col++) {
                    int in_row = out_row * cfg.stride + filter_row - cfg.padding;
                    int in_col = out_col * cfg.stride + filter_col - cfg.padding;

                    if (in_row >= 0 && in_row < cfg.height && in_col >= 0 && in_col < cfg.width) {
                        int input_index = in_row * cfg.width + in_col;
                        int filter_index = filter_row * cfg.filter_size + filter_col;
                        accumulator += input[input_index] * filter[filter_index];
                    }
                }
            }

            output[out_row * out_w + out_col] = accumulator;
        }
    }
}

bool compare_outputs(const float* gpu_output,
                     const float* cpu_output,
                     int output_elems) {
    const float tolerance = 1e-3f;

    for (int i = 0; i < output_elems; i++) {
        float diff = fabsf(gpu_output[i] - cpu_output[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: GPU %.6f, CPU %.6f, diff %.6f\n",
                   i, gpu_output[i], cpu_output[i], diff);
            return false;
        }
    }

    return true;
}

__global__ void conv2d_naive_kernel(const float* input,
                                    const float* filter,
                                    float* output,
                                    Conv2DConfig cfg) { 
    //Thread output coordinates:
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    //output size:
    int out_h = (cfg.height - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;
    int out_w = (cfg.width - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;

    //Boundary guards:
    if (out_row >= out_h || out_col >= out_w) return;

    float accumulator = 0.0f;

    for (int filter_row = 0; filter_row < cfg.filter_size; filter_row++) {
        for (int filter_col = 0; filter_col < cfg.filter_size; filter_col++) {
            int in_row = out_row * cfg.stride + filter_row - cfg.padding;
            int in_col = out_col * cfg.stride + filter_col - cfg.padding;

            //check padding and then compute dot product:
            if (in_row >= 0 && in_row < cfg.height && in_col >= 0 && in_col < cfg.width) {
                int input_index = in_row * cfg.width + in_col;
                int filter_index = filter_row * cfg.filter_size + filter_col;
                accumulator += input[input_index] * filter[filter_index];
            }
        }
    }
    //write dot product value into correct index in output array
    int output_index = out_row * out_w + out_col;
    output[output_index] = accumulator;
}

void launch_naive_conv2d(const float* d_input,
                         const float* d_filter,
                         float* d_output,
                         Conv2DConfig cfg) {
    int out_h = (cfg.height - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;
    int out_w = (cfg.width - cfg.filter_size + 2 * cfg.padding) / cfg.stride + 1;

    dim3 blockDim(16,16);
    dim3 gridDim((out_w + blockDim.x -1) / blockDim.x, (out_h + blockDim.y - 1) / blockDim.y);
    
    conv2d_naive_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, cfg);
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
    float* h_reference = new float[output_elems];

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
    launch_naive_conv2d(d_input, d_filter, d_output, cfg);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    launch_naive_conv2d(d_input, d_filter, d_output, cfg);
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
    conv2d_cpu_reference(h_input, h_filter, h_reference, cfg);
    printf("Correctness: %s\n", compare_outputs(h_output, h_reference, output_elems) ? "PASS" : "FAIL");

    /**
    for (int r = 0; r < out_h; ++r) {
          for (int c = 0; c < out_w; ++c) {
              printf("%6.1f ", h_output[r * out_w + c]);
          }
          printf("\n");
    }**/

    delete[] h_input; delete[] h_filter; delete[] h_output; delete[] h_reference;
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
