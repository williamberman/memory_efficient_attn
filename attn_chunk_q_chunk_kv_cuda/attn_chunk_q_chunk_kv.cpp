#include <torch/extension.h>

void attn_chunk_q_chunk_kv_kernel(
    float *q, float *k, float *v, float *out, 
    int b, int h, int q_seq_len, int d, 
    int kv_seq_len,
    int q_b_stride, int q_h_stride, int qkv_seq_stride,
    int kv_b_stride, int kv_h_stride
);

at::Tensor attn_chunk_q_chunk_kv_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(k.is_cuda());
    TORCH_CHECK(v.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kFloat32);
    TORCH_CHECK(k.dtype() == torch::kFloat32);
    TORCH_CHECK(v.dtype() == torch::kFloat32);
    TORCH_CHECK(q.is_contiguous());
    TORCH_CHECK(k.is_contiguous());
    TORCH_CHECK(v.is_contiguous());

    auto out = torch::zeros_like(q);

    auto q_shape = q.sizes();
    auto kv_shape = k.sizes();
    auto q_strides = q.strides();
    auto kv_strides = k.strides();

    attn_chunk_q_chunk_kv_kernel(
        static_cast<float *>(q.data_ptr()), 
        static_cast<float *>(k.data_ptr()), 
        static_cast<float *>(v.data_ptr()),
        static_cast<float *>(out.data_ptr()),

        static_cast<int>(q_shape[0]),
        static_cast<int>(q_shape[1]),
        static_cast<int>(q_shape[2]),
        static_cast<int>(q_shape[3]),

        kv_shape[2],

        static_cast<int>(q_strides[0]),
        static_cast<int>(q_strides[1]),
        static_cast<int>(q_strides[2]),

        static_cast<int>(kv_strides[0]),
        static_cast<int>(kv_strides[1])
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attn_chunk_q_chunk_kv_cuda", &attn_chunk_q_chunk_kv_cuda, "attn_chunk_q_chunk_kv_cuda");
}