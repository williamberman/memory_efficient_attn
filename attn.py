import numpy as np
from scipy.special import softmax


def single_query_vector_attn(q, k, v):
    d = q.shape[0]
    qk = np.einsum("d, kd -> k", q, k) / d**0.5
    sm_qk = softmax(qk)
    return np.einsum("k, kd -> d", sm_qk, v)


def single_query_vector_attn_lazy_softmax(q, k, v):
    d = q.shape[0]
    qk = np.einsum("d, kd -> k", q, k) / d**0.5
    qk_max = np.max(qk, -1, keepdims=True)
    sm_num = np.exp(qk - qk_max)
    sm_denom = np.sum(sm_num, keepdims=True)
    num = np.einsum("k, kd -> d", sm_num, v)
    return num / sm_denom


def attn_lazy_softmax(q, k, v):
    d = q.shape[-1]
    qk = np.einsum("bhqd, bhkd -> bhqk", q, k) / d**0.5
    qk_max = np.max(qk, -1, keepdims=True)
    sm_num = np.exp(qk - qk_max)
    sm_denom = np.sum(sm_num, -1, keepdims=True)
    num = np.einsum("bhqk, bhkd -> bhqd", sm_num, v)
    return num / sm_denom


def attn_lazy_softmax_components(q, k, v):
    d = q.shape[-1]
    qk = np.einsum("bhqd, bhkd -> bhqk", q, k) / d**0.5
    qk_max = np.max(qk, -1, keepdims=True)
    sm_num = np.exp(qk - qk_max)
    sm_denom = np.sum(sm_num, -1, keepdims=True)
    num = np.einsum("bhqk, bhkd -> bhqd", sm_num, v)
    return num, sm_denom, qk_max


def attn_split_kv(q, k, v):
    # split along the sequence length
    k_1, k_2 = np.split(k, 2, 2)
    v_1, v_2 = np.split(v, 2, 2)

    num_1, denom_1, max_1 = attn_lazy_softmax_components(q, k_1, v_1)
    num_2, denom_2, max_2 = attn_lazy_softmax_components(q, k_2, v_2)

    # renormalize and complete weighted average terms
    max = np.maximum(max_1, max_2)
    num = num_1 * np.exp(max_1 - max) + num_2 * np.exp(max_2 - max)
    denom = denom_1 * np.exp(max_1 - max) + denom_2 * np.exp(max_2 - max)

    return num / denom


def attn_chunk_kv(q, k, v, kv_chunk_seq_len):
    d = q.shape[-1]

    # split along the sequence length
    kv_seq_len = k.shape[-2]
    kv_num_chunks = kv_seq_len // kv_chunk_seq_len
    k_chunks = np.split(k, kv_num_chunks, 2)
    v_chunks = np.split(v, kv_num_chunks, 2)

    b, h, q_seq_len, d = q.shape
    nums = np.empty((kv_num_chunks, b, h, q_seq_len, d))
    denoms = np.empty((kv_num_chunks, b, h, q_seq_len, 1))
    maxes = np.empty((kv_num_chunks, b, h, q_seq_len, 1))

    for i, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
        num, denom, max = attn_lazy_softmax_components(q, k_chunk, v_chunk)
        nums[i] = num ; denoms[i] = denom ; maxes[i] = max  # fmt: skip

    # renormalize and complete weighted average terms
    max = maxes.max(0, keepdims=True)
    num = (nums * np.exp(maxes - max)).sum(0)
    denom = (denoms * np.exp(maxes - max)).sum(0)

    return num / denom


def attn_chunk_q_chunk_kv(q, k, v, q_chunk_seq_len, kv_chunk_seq_len):
    b, h, q_seq_len, d = q.shape

    # split along the sequence length
    q_num_chunks = q_seq_len // q_chunk_seq_len
    q_chunks = np.split(q, q_num_chunks, 2)

    # split along the sequence length
    kv_seq_len = k.shape[-2]
    kv_num_chunks = kv_seq_len // kv_chunk_seq_len
    k_chunks = np.split(k, kv_num_chunks, 2)
    v_chunks = np.split(v, kv_num_chunks, 2)

    # pre-allocate output value
    out = np.empty_like(q)

    for q_idx, q_chunk in enumerate(q_chunks):
        # allocate aggregate statistics for a given q chunk
        # Rabe and Staats just assumes that these are written to DRAM
        nums = np.empty((kv_num_chunks, b, h, q_chunk_seq_len, d))
        denoms = np.empty((kv_num_chunks, b, h, q_chunk_seq_len, 1))
        maxes = np.empty((kv_num_chunks, b, h, q_chunk_seq_len, 1))

        for kv_idx, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
            num, denom, max = attn_lazy_softmax_components(q_chunk, k_chunk, v_chunk)
            nums[kv_idx] = num ; denoms[kv_idx] = denom ; maxes[kv_idx] = max  # fmt: skip

        # renormalize and complete weighted average terms for the q chunk
        max = maxes.max(0, keepdims=True)
        num = (nums * np.exp(maxes - max)).sum(0)
        denom = (denoms * np.exp(maxes - max)).sum(0)

        # incrementally write into output value for each q chunk
        q_store_start = q_idx * q_chunk_seq_len
        q_store_indices = slice(q_store_start, q_store_start + q_chunk_seq_len)
        out[:, :, q_store_indices, :] = num / denom

    return out


def attn_chunk_kv_chunk_q(q, k, v, q_chunk_seq_len, kv_chunk_seq_len):
    q_seq_len = q.shape[-2]
    q_num_chunks = q_seq_len // q_chunk_seq_len
    q_chunks = np.split(q, q_num_chunks, 2)

    kv_seq_len = k.shape[-2]
    kv_num_chunks = kv_seq_len // kv_chunk_seq_len
    k_chunks = np.split(k, kv_num_chunks, 2)
    v_chunks = np.split(v, kv_num_chunks, 2)

    # softmax statistics for each kv, q chunk pair
    b, h, q_seq_len, d = q.shape
    nums = np.empty((kv_num_chunks, b, h, q_seq_len, d))
    denoms = np.empty((kv_num_chunks, b, h, q_seq_len, 1))
    maxes = np.empty((kv_num_chunks, b, h, q_seq_len, 1))

    # outer loop is now kv
    for kv_idx, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
        for q_idx, q_chunk in enumerate(q_chunks):
            num, denom, max = attn_lazy_softmax_components(q_chunk, k_chunk, v_chunk)

            # write to softmax statistics for each kv, q chunk pair
            q_store_start = q_idx * q_chunk_seq_len
            q_store_indices = slice(q_store_start, q_store_start + q_chunk_seq_len)
            nums[kv_idx, :, :, q_store_indices, :] = num
            denoms[kv_idx, :, :, q_store_indices, :] = denom
            maxes[kv_idx, :, :, q_store_indices, :] = max

    # renormalize and complete weighted average terms
    max = maxes.max(0, keepdims=True)
    num = (nums * np.exp(maxes - max)).sum(0)
    denom = (denoms * np.exp(maxes - max)).sum(0)

    return num / denom


def attn_chunk_kv_chunk_q_incremental(q, k, v, q_chunk_seq_len, kv_chunk_seq_len):
    q_seq_len = q.shape[-2]
    q_num_chunks = q_seq_len // q_chunk_seq_len
    q_chunks = np.split(q, q_num_chunks, 2)

    kv_seq_len = k.shape[-2]
    kv_num_chunks = kv_seq_len // kv_chunk_seq_len
    k_chunks = np.split(k, kv_num_chunks, 2)
    v_chunks = np.split(v, kv_num_chunks, 2)

    # softmax statistics _just_ for each q chunk
    b, h, q_seq_len, d = q.shape
    nums = np.zeros((b, h, q_seq_len, d))
    denoms = np.zeros((b, h, q_seq_len, 1))
    maxes = np.full((b, h, q_seq_len, 1), -np.inf)

    # used for clarity but not necessary, can just write results into `nums`
    # array once all incremental statistics are computed
    out = np.empty_like(q)

    for kv_idx, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
        for q_idx, q_chunk in enumerate(q_chunks):
            num, denom, max = attn_lazy_softmax_components(q_chunk, k_chunk, v_chunk)

            q_store_start = q_idx * q_chunk_seq_len
            q_store_indices = slice(q_store_start, q_store_start + q_chunk_seq_len)

            # renormalize and incrementally update weighted average terms
            num_acc = nums[:, :, q_store_indices, :]
            denom_acc = denoms[:, :, q_store_indices, :]
            max_acc = maxes[:, :, q_store_indices, :]

            new_max = np.maximum(max, max_acc)

            num_acc = num * np.exp(max - new_max) + num_acc * np.exp(max_acc - new_max)
            denom_acc = denom * np.exp(max - new_max) + denom_acc * np.exp(max_acc - new_max)  # fmt: skip

            if kv_idx != kv_num_chunks - 1:
                nums[:, :, q_store_indices, :] = num_acc
                denoms[:, :, q_store_indices, :] = denom_acc
                maxes[:, :, q_store_indices, :] = new_max
            else:
                out[:, :, q_store_indices, :] = num_acc / denom_acc

    return out
