//! CUDA kernel source code
//!
//! These kernels are compiled at build time and loaded at runtime.

/// Layer normalization CUDA kernel source
pub const LAYER_NORM_KERNEL: &str = r#"
extern "C" __global__ void layer_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float eps,
    const int hidden_size,
    const int batch_size
) {
    // Each block handles one row (one sample in the batch)
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    // Shared memory for reduction
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + blockDim.x;

    // Compute local sum and sum of squares
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Compute mean and variance
    float mean = s_sum[0] / (float)hidden_size;
    float variance = s_sum_sq[0] / (float)hidden_size - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    // Normalize and apply weight/bias
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (row_input[i] - mean) * inv_std;
        row_output[i] = normalized * weight[i] + bias[i];
    }
}

extern "C" __global__ void layer_norm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const float eps,
    const int hidden_size,
    const int batch_size
) {
    // Each block handles one row
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* row_input = input + row * hidden_size;
    __half* row_output = output + row * hidden_size;

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sum_sq = shared + blockDim.x;

    // Compute in float for numerical stability
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)hidden_size;
    float variance = s_sum_sq[0] / (float)hidden_size - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_input[i]);
        float normalized = (val - mean) * inv_std;
        float w = __half2float(weight[i]);
        float b = __half2float(bias[i]);
        row_output[i] = __float2half(normalized * w + b);
    }
}
"#;

/// RMS normalization kernel (used by some models)
pub const RMS_NORM_KERNEL: &str = r#"
extern "C" __global__ void rms_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const float eps,
    const int hidden_size,
    const int batch_size
) {
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    extern __shared__ float s_sum_sq[];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];
        local_sum_sq += val * val;
    }

    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float rms = rsqrtf(s_sum_sq[0] / (float)hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_output[i] = row_input[i] * rms * weight[i];
    }
}
"#;
