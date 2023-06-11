from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='attn_chunk_q_chunk_kv_cuda',
      ext_modules=[CUDAExtension('attn_chunk_q_chunk_kv_cuda', ['attn_chunk_q_chunk_kv.cpp', 'attn_chunk_q_chunk_kv_kernel.cu'])],
      cmdclass={'build_ext': BuildExtension})
