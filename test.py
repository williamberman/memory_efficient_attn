import torch
import numpy as np

from torch.nn import functional as F

from attn import *


def test_single_query_vector_attn(d, kv_seq_len):
    q = np.random.random((d))
    k = np.random.random((kv_seq_len, d))
    v = k

    out = single_query_vector_attn(q, k, v)

    q_torch = torch.from_numpy(q)[None, :]
    k_torch = torch.from_numpy(k)
    v_torch = torch.from_numpy(v)

    out_torch = F.scaled_dot_product_attention(q_torch, k_torch, v_torch)

    assert np.allclose(out_torch.detach().numpy(), out)


def test_single_query_vector_attn_lazy_softmax(d, kv_seq_len):
    q = np.random.random((d))
    k = np.random.random((kv_seq_len, d))
    v = k

    out = single_query_vector_attn_lazy_softmax(q, k, v)

    q_torch = torch.from_numpy(q)[None, :]
    k_torch = torch.from_numpy(k)
    v_torch = torch.from_numpy(v)

    out_torch = F.scaled_dot_product_attention(q_torch, k_torch, v_torch)

    assert np.allclose(out_torch.detach().numpy(), out)


def test_attn(
    b, h, q_seq_len, d, kv_seq_len, q_chunk_seq_len=None, kv_chunk_seq_len=None, f=None
):
    q = np.random.random((b, h, q_seq_len, d))
    k = np.random.random((b, h, kv_seq_len, d))
    v = k

    opt_args = {}
    if q_chunk_seq_len is not None:
        opt_args["q_chunk_seq_len"] = q_chunk_seq_len
    if kv_chunk_seq_len is not None:
        opt_args["kv_chunk_seq_len"] = kv_chunk_seq_len

    out = f(q, k, v, **opt_args)

    q_torch = torch.from_numpy(q)
    k_torch = torch.from_numpy(k)
    v_torch = torch.from_numpy(v)

    out_torch = F.scaled_dot_product_attention(q_torch, k_torch, v_torch)

    assert np.allclose(out_torch.detach().numpy(), out)


def test_attn_chunk_q_chunk_kv_cuda(b, h, q_seq_len, d, kv_seq_len):
    from attn_chunk_q_chunk_kv_cuda import attn_chunk_q_chunk_kv_cuda

    q = torch.rand((b, h, q_seq_len, d), device="cuda")
    k = torch.rand((b, h, kv_seq_len, d), device="cuda")
    v = k

    out = attn_chunk_q_chunk_kv_cuda(q, k, v)

    out_ = F.scaled_dot_product_attention(q, k, v)

    assert torch.allclose(out_, out)


def test_all():
    for test in [
        test_single_query_vector_attn,
        test_single_query_vector_attn_lazy_softmax,
    ]:
        print(test.__name__)
        for d in range(1, 6):
            for kv_seq_len in range(1, 6):
                test(d, kv_seq_len)

    print(attn_lazy_softmax.__name__)
    for b in range(1, 3):
        for h in range(1, 3):
            for q_seq_len in range(1, 6):
                for d in range(1, 6):
                    for kv_seq_len in range(1, 6):
                        test_attn(b, h, q_seq_len, d, kv_seq_len, f=attn_lazy_softmax)

    print(attn_split_kv.__name__)
    for b in range(1, 3):
        for h in range(1, 3):
            for q_seq_len in range(1, 6):
                for d in range(1, 6):
                    for kv_seq_len in [2, 4, 6]:
                        test_attn(b, h, q_seq_len, d, kv_seq_len, f=attn_split_kv)

    print(attn_chunk_kv.__name__)
    for b in range(1, 3):
        for h in range(1, 3):
            for q_seq_len in range(1, 3):
                for d in range(1, 6):
                    for kv_chunk_seq_len in range(1, 3):
                        for kv_seq_len_multiple in range(1, 6):
                            kv_seq_len = kv_chunk_seq_len * kv_seq_len_multiple
                            test_attn(
                                b,
                                h,
                                q_seq_len,
                                d,
                                kv_seq_len,
                                kv_chunk_seq_len=kv_chunk_seq_len,
                                f=attn_chunk_kv,
                            )

    for f in [
        attn_chunk_q_chunk_kv,
        attn_chunk_kv_chunk_q,
        attn_chunk_kv_chunk_q_incremental,
    ]:
        print(f.__name__)
        for b in range(1, 3):
            for h in range(1, 3):
                for q_chunk_seq_len in range(1, 3):
                    for q_seq_len_multiple in range(1, 3):
                        for d in range(1, 6):
                            for kv_chunk_seq_len in range(1, 3):
                                for kv_seq_len_multiple in range(1, 3):
                                    q_seq_len = q_chunk_seq_len * q_seq_len_multiple
                                    kv_seq_len = kv_chunk_seq_len * kv_seq_len_multiple
                                    test_attn(
                                        b,
                                        h,
                                        q_seq_len,
                                        d,
                                        kv_seq_len,
                                        q_chunk_seq_len,
                                        kv_chunk_seq_len,
                                        f=f,
                                    )

    if torch.cuda.is_available():
        from attn_chunk_q_chunk_kv_cuda import attn_chunk_q_chunk_kv_cuda

        print(attn_chunk_q_chunk_kv_cuda.__name__)

        for b in range(1, 3):
            for h in range(1, 3):
                for q_seq_len in range(1, 6):
                    for d in range(1, 6):
                        for kv_seq_len in range(1, 6):
                            test_attn_chunk_q_chunk_kv_cuda(
                                b, h, q_seq_len, d, kv_seq_len
                            )
    else:
        print("cuda not available, skipping cuda tests")


if __name__ == "__main__":
    test_all()
