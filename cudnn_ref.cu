#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_ops.h>
#include <stdio.h>
#include <math.h>

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

int main(int argc, char** argv) {
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

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t input_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(input_desc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               1, 1, cfg.height, cfg.width);

    cudnnSetFilter4dDescriptor(filter_desc,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               1, 1,
                               cfg.filter_size, cfg.filter_size);

    cudnnSetConvolution2dDescriptor(conv_desc,
                                    cfg.padding, cfg.padding,
                                    cfg.stride, cfg.stride,
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    int out_n, out_c, cudnn_out_h, cudnn_out_w;
    cudnnGetConvolution2dForwardOutputDim(conv_desc,
                                          input_desc,
                                          filter_desc,
                                          &out_n,
                                          &out_c,
                                          &cudnn_out_h,
                                          &cudnn_out_w);

    cudnnSetTensor4dDescriptor(output_desc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               out_n, out_c, cudnn_out_h, cudnn_out_w);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_desc,
                                            filter_desc,
                                            conv_desc,
                                            output_desc,
                                            algo,
                                            &workspace_bytes);

    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        cudaMalloc(&d_workspace, workspace_bytes);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //warmup run:
    cudnnConvolutionForward(handle,
                            &alpha,
                            input_desc,
                            d_input,
                            filter_desc,
                            d_filter,
                            conv_desc,
                            algo,
                            d_workspace,
                            workspace_bytes,
                            &beta,
                            output_desc,
                            d_output);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cudnnStatus_t status = cudnnConvolutionForward(handle,
                                                   &alpha,
                                                   input_desc,
                                                   d_input,
                                                   filter_desc,
                                                   d_filter,
                                                   conv_desc,
                                                   algo,
                                                   d_workspace,
                                                   workspace_bytes,
                                                   &beta,
                                                   output_desc,
                                                   d_output);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    if (status != CUDNN_STATUS_SUCCESS) {
        printf("cuDNN error: %s\n", cudnnGetErrorString(status));
        return 1;
    }

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

    delete[] h_input;
    delete[] h_filter;
    delete[] h_output;
    delete[] h_reference;
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    if (d_workspace != nullptr) {
        cudaFree(d_workspace);
    }
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
