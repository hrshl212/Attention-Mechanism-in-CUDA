import torch
import attention_cuda
import torch.nn.functional as F
import time

# Define input sizes
batch_size, seq_len, d_model, n_heads = 16, 128, 512, 8
d_k = d_model // n_heads    #64

# Create inputs on GPU
Q = torch.randn(batch_size, seq_len, d_k, device="cuda", dtype=torch.float32)
K = torch.randn(batch_size, seq_len, d_k, device="cuda", dtype=torch.float32)
V = torch.randn(batch_size, seq_len, d_k, device="cuda", dtype=torch.float32)

start_time = time.time()
# Run CUDA optimized attention
attn_output = attention_cuda.attention_forward(Q, K, V)
end_time = time.time()

attn_output2 = F.scaled_dot_product_attention(Q, K, V)
end_time2 = time.time()

elapsed_time1 = end_time - start_time
elapsed_time2 = end_time2 - end_time

print(torch.equal(attn_output, attn_output2))


import pdb
pdb.set_trace()

print("Attention output shape:", attn_output.shape)  # Should be (batch_size, seq_len, d_k)
print(f"Elapsed time1: {elapsed_time1} sec")
print(f"Elapsed time2: {elapsed_time2} sec")
