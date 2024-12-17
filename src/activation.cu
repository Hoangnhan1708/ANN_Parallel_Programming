#include "activation.cuh"
#include "error_checking.cuh"

__global__ void ReLuKernel(float* input,
                           const int size);


// Changed
__global__ void SoftmaxKernel(const int output_size,
                                 const int batch_size,
                                 float* values);

void ReLU::operator()(const int output_size,
                      const int batch_size,
                      float* d_value) {
    const int total_size = batch_size * output_size;
    const int threadsPerBlock = 256;
    const int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    ReLuKernel<<<numBlocks, threadsPerBlock>>>(d_value, total_size);
    CHECK_LAST_CUDA_ERROR();
}


// Changed
void SoftMax::operator()(const int batch_size,
    const int output_size,
    float* d_value) {
    const int total_size = batch_size * output_size;
    const int threadsPerBlock = 256;
    const int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    SoftmaxKernel<<<numBlocks, threadsPerBlock>>>(output_size, batch_size, d_value);
    CHECK_LAST_CUDA_ERROR();
}


__global__ void ReLuKernel(float* input,
                           const int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        input[idx] = fmaxf(0, input[idx]);
    }
}


// Changed
__global__ void SoftmaxKernel(const int output_size,
                              const int batch_size,
                              float* values) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size)
    return;

    // Calculate max for numerical stability (log-sum-exp trick for overflow protection)
    float maxInput = -INFINITY;
    for (int j = 0; j < output_size; ++j) {
        maxInput = fmaxf(maxInput, values[idx * output_size + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < output_size; ++j) {
        // Subtract maxInput for numerical stability to avoid overflow issues
        sum += expf(values[idx * output_size + j] - maxInput);
    }

    // Calculate Softmax for each element in the batch that this thread should process
    for (int j = 0; j < output_size; ++j) {
        // Apply Softmax formula: exp(input) / sum(exp(input))
        values[idx * output_size + j] = expf(values[idx * output_size + j] - maxInput) / sum;
    }
}