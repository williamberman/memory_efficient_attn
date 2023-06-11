Collection of toy implementations of memory efficient attention.

All numpy implementations are in `attn.py`, and there's a cuda implementation in `attn_chunk_q_chunk_kv_cuda/attn_chunk_q_chunk_kv_kernel.cu`

##### Building the cuda implementation
```sh
cd attn_chunk_q_chunk_kv_cuda
python setup.py install
```

##### Running the tests
No formal testing framework, just run 

```sh
python test.py
````

If no assertions are thrown, tests pass :)
