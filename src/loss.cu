#include "error_checking.cuh"
#include "loss.cuh"
#include <cuda_runtime.h>

__global__ void ComputeDzLastLayerKernel(const float* __restrict__ log_predictions,
                                         const int* __restrict__ labels,
                                         float* __restrict__ dZ,
                                         const int numClasses,
                                         const int batchSize);

__global__ void CrossEntropyLossKernel(const float* __restrict__ predictions,
                                       const int* __restrict__ labels,
                                       float* __restrict__ loss,
                                       const int numClasses,
                                       const int batchSize);


__global__ void SoftmaxCrossEntropyLossKernel(const float* __restrict__ values,
                                                 float* __restrict__ predictions,
                                                 const int* __restrict__ labels,
                                                 float* __restrict__ loss,
                                                 float* __restrict__ dZ,
                                                 const int output_size,
                                                 const int batch_size);

CrossEntropyLoss::CrossEntropyLoss(const int num_classes, const int batch_size) : _num_classes(num_classes),
                                                                                  _batch_size(batch_size) {
    CHECK_CUDA_ERROR(cudaMalloc(&_d_dZ, sizeof(float) * num_classes * batch_size));
    CHECK_CUDA_ERROR(cudaMalloc(&_d_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&_d_predictions, sizeof(float) * num_classes * batch_size));
}

CrossEntropyLoss::~CrossEntropyLoss() {
    CHECK_CUDA_ERROR(cudaFree(_d_dZ));
    CHECK_CUDA_ERROR(cudaFree(_d_loss));
    CHECK_CUDA_ERROR(cudaFree(_d_predictions));
}

float CrossEntropyLoss::operator()(const float* d_values, const int* d_labels) {
    float loss;
    const int threadsPerBlock = 50;
    const int blocksPerGrid = (_batch_size + threadsPerBlock - 1) / threadsPerBlock;
    CHECK_CUDA_ERROR(cudaMemset(_d_loss, 0, sizeof(float)));
    // Changed
    SoftmaxCrossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock>>>(d_values,
                                                                        _d_predictions,
                                                                        d_labels,
                                                                        _d_loss,
                                                                        _d_dZ,
                                                                        _num_classes,
                                                                        _batch_size);

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaMemcpy(&loss, _d_loss, sizeof(float), cudaMemcpyDeviceToHost));

    return loss / _batch_size;
}

float* CrossEntropyLoss::Backward() {
    return _d_dZ;
}

__global__ void CrossEntropyLossKernel(const float* __restrict__ predictions,
                                       const int* __restrict__ labels,
                                       float* __restrict__ loss,
                                       const int numClasses,
                                       const int batch_size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size)
        return;

    const int label = labels[idx];
    const float prediction = predictions[idx * numClasses + label];
    atomicAdd(loss, -prediction);
}


__global__ void SoftmaxCrossEntropyLossKernel(const float* __restrict__ values,
                                              float* __restrict__ predictions,
                                              const int* __restrict__ labels,
                                              float* __restrict__ loss,
                                              float* __restrict__ dZ,
                                              const int output_size,
                                              const int batch_size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size)
        return;

    float maxInput = -INFINITY;

    // Tìm giá trị lớn nhất trong logits cho mỗi ví dụ (chống hiện tượng số học nhỏ)
#pragma unroll
    for (int j = 0; j < output_size; ++j) {
        maxInput = fmaxf(maxInput, values[idx * output_size + j]);
    }

    float sum = 0.0f;

    // Tính tổng các giá trị e^logits sau khi trừ đi maxInput (cho ổn định số học)
#pragma unroll
    for (int j = 0; j < output_size; ++j) {
        sum += expf(values[idx * output_size + j] - maxInput);
    }

    const int label = labels[idx];
    float softmax_value = 0.0f;

    // Tính giá trị softmax và loss
#pragma unroll
    for (int j = 0; j < output_size; ++j) {
        float softmax = expf(values[idx * output_size + j] - maxInput) / sum;
        predictions[idx * output_size + j] = softmax;  // lưu xác suất softmax

        // Xác định softmax cho nhãn đúng và tính loss
        if (j == label) {
            softmax_value = softmax;
        }

        // Tính đạo hàm dZ cho backpropagation
        dZ[idx * output_size + j] = softmax - (j == label ? 1.0f : 0.0f);
    }

    // Cập nhật giá trị loss (tính Cross-Entropy)
    atomicAdd(loss, -logf(softmax_value + 1e-8f)); // tránh log(0) bằng cách thêm epsilon
}

__global__ void ComputeDzLastLayerKernel(const float* __restrict__ log_predictions,
                                         const int* __restrict__ labels,
                                         float* __restrict__ dZ,
                                         const int numClasses,
                                         const int batchSize) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize)
        return;

    for (int i = 0; i < numClasses; i++) {
        const float softmax_output = expf(log_predictions[idx * numClasses + i]); // convert log_softmax to softmax
        dZ[idx * numClasses + i] = softmax_output - (i == labels[idx] ? 1.0f : 0.0f);
    }
}