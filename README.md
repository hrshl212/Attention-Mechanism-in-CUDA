# Attention-Mechanism-in-CUDA

In this repository, I have implemented the attention mechanism in C++ and CUDA and integrated it with PyTorch. The attention_cuda.cu file contains the CUDA kernels. Shared memory has been utilized to reduce global memory accesses and also to boost coalesced memory access. The CUDA kernel is compiled using setup.py file. Run the following command to compile and install the CUDA extension.

After compiling the CUDA code, the optimized scaled dot-product attention can be used in PyTorch. The custom implementation has been found to be 2x faster than the PyTorch's inbuilt scaled_dot_product_attention implementation. For the dimensions in the code, the PyTorch implementation took a time of 0.056s whereas the custom implementation took a time of 0.027s.
