#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void attn_chunk_q_chunk_kv_kernel_(
    float *q, float *k, float *v, float *out, 
    int kv_seq_len,
    int q_b_stride, int q_h_stride, int qkv_seq_stride,
    int kv_b_stride, int kv_h_stride
) {
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int q_seq_idx = blockIdx.z;
    int d_idx = threadIdx.x;
    int d = blockDim.x;

    extern __shared__ float smem[];
    float *q_smem = smem;
    float *k_smem = &smem[d];
    float *v_smem = &smem[2*d];
    float *qk_smem = &smem[3*d];
    float *num = &smem[4*d];
    float *denom = &smem[5*d];
    float *max = &smem[5*d+1];

    // Load q
    int q_idx = b_idx * q_b_stride + h_idx * q_h_stride + q_seq_idx * qkv_seq_stride + d_idx;
    q_smem[d_idx] = q[q_idx];
    __syncthreads();

    for (int kv_seq_idx = 0; kv_seq_idx < kv_seq_len; ++kv_seq_idx) {
        // Load kv
        int kv_idx = b_idx * kv_b_stride + h_idx * kv_h_stride + kv_seq_idx * qkv_seq_stride + d_idx;
        k_smem[d_idx] = k[kv_idx];
        v_smem[d_idx] = v[kv_idx];
        __syncthreads();

        // Compute q*k pointwise
        qk_smem[d_idx] = q_smem[d_idx] * k_smem[d_idx];
        __syncthreads();

        // Compute rest of qk inner product - repeated amongst whole block
        float qk = 0;
        for (int i = 0; i < d; ++i) {
            qk += qk_smem[i];
        }

        // Scale
        qk /= std::sqrt(static_cast<float>(d));

        if (kv_seq_idx == 0) {
            // First step of loop, normalized sm_num and denom must be 1, no previous max
            *max = qk;
            num[d_idx] = v_smem[d_idx];
            *denom = 1;
        } else {
            // Compute and update max
            float max_ = *max;
            float new_max = qk > max_ ? qk : max_;
            *max = new_max;

            float sm_num = std::exp(qk - new_max);
            float weighted_avg_num = sm_num * v_smem[d_idx];

            // Renormalize and update num
            float num_ = num[d_idx];
            num_ *= std::exp(max_ - new_max);
            num_ += weighted_avg_num;
            num[d_idx] = num_;

            // Renormalize and update denom
            float denom_ = *denom;
            denom_ *= std::exp(max_ - new_max);
            denom_ += sm_num;
            *denom = denom_;
        }

        __syncthreads();
    }

    // Compute final value and write to output
    float out_ = num[d_idx] / *denom;
    out[q_idx] = out_;
}

void attn_chunk_q_chunk_kv_kernel(
    float *q, float *k, float *v, float *out, 
    int b, int h, int q_seq_len, int d, 
    int kv_seq_len,
    int q_b_stride, int q_h_stride, int qkv_seq_stride,
    int kv_b_stride, int kv_h_stride
) {
    dim3 grid(b, h, q_seq_len);
    int block = d;
    int smem_size = (5 * d + 2)*sizeof(float);

    attn_chunk_q_chunk_kv_kernel_<<<grid, block, smem_size>>>(
        q, k, v, out, 
        kv_seq_len,
        q_b_stride, q_h_stride, qkv_seq_stride,
        kv_b_stride, kv_h_stride
    );
}
