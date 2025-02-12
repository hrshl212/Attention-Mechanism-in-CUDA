#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32  // Define block size for matrix multiplication

__global__ void key_query_multiplication(
    const float* Q, 
    const float* K,  
    float* QK_product, 
    int batch_size, 
    int seq_len, 
    int d_k) {

    int batch_idx = blockIdx.z; // Batch index (0 to 15)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Sequence index (0 to 511)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Seq index (0 to 511)

    if (row < seq_len && col < seq_len){

        // Calculate the dot product for each head at each position
        // Load Q, K matrices into shared memory
        __shared__ float shared_Q[32][32];  // Using 32x32 shared memory for Q
        __shared__ float shared_K[32][32];  // Using 32x32 shared memory for K

        // Step 1: Compute dot product between Q and K
        float dot_product = 0.0f;
        for (int i = 0; i < (d_k + BLOCK_SIZE-1) / BLOCK_SIZE; i++) {
            shared_Q[threadIdx.y][threadIdx.x] = Q[batch_idx * seq_len * d_k + row * d_k + (i * 32+threadIdx.x)];
            shared_K[threadIdx.y][threadIdx.x] = K[batch_idx * seq_len * d_k + row * d_k + (i * 32+threadIdx.x)];
            __syncthreads();

            for (int n = 0; n < 32; n++) {
                int idy = i*BLOCK_SIZE + n;
                if (idy < d_k){
                    dot_product += shared_Q[threadIdx.y][n] * shared_K[threadIdx.x][n];
                }
            }        
            __syncthreads();
        }    

        // Scale the dot product by the square root of the key dimension
        dot_product /= sqrtf(d_k);
        QK_product[batch_idx * seq_len * d_k + row * seq_len + col] = dot_product;
    }

    __syncthreads();
    // Step 2: Apply Softmax row-wise
    if (col < seq_len && row < seq_len) {
        // Compute max value in row for numerical stability
        float max_val = -1e20;
        for (int j = 0; j < seq_len; j++) {
            max_val = fmaxf(max_val, QK_product[batch_idx * seq_len * d_k + row * seq_len + j]);
        }

        __syncthreads();
        // Compute exponentials and sum
        float exp_sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            QK_product[batch_idx * seq_len * d_k + row * seq_len + j] = expf(QK_product[batch_idx * seq_len * d_k + row * seq_len + j] - max_val);
            exp_sum += QK_product[batch_idx * seq_len * d_k + row * seq_len + j];
        }

        __syncthreads();
        // Normalize by sum
        for (int j = 0; j < seq_len; j++) {
            QK_product[batch_idx * seq_len * d_k + row * seq_len + j] /= exp_sum;
        }
    }
}

__global__ void scaled_dot_product_attention(
    const float* QK_product, 
    const float* V, 
    float* output, 
    int batch_size, 
    int seq_len, 
    int d_k) {

    // Shared memory tiles
    __shared__ float QK_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float V_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate row and column indices
    int batch_idx = blockIdx.z; // Batch index (0 to 15)
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;  // Accumulate the result

    // Loop over tiles of A and B
    for (int t = 0; t < (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load A and B tiles into shared memory (ensure within bounds)
        if (row < seq_len && (t * BLOCK_SIZE + threadIdx.x) < seq_len)
            QK_tile[threadIdx.y][threadIdx.x] = QK_product[batch_idx * seq_len * d_k + row * seq_len + t * BLOCK_SIZE + threadIdx.x];
        else
            QK_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < d_k && (t * BLOCK_SIZE + threadIdx.y) < seq_len)
            V_tile[threadIdx.y][threadIdx.x] = V[batch_idx * seq_len * d_k + (t * BLOCK_SIZE + threadIdx.y) * d_k + col];
        else
            V_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Synchronize threads

        // Compute partial sum for C[row, col]
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += QK_tile[threadIdx.y][k] * V_tile[k][threadIdx.x];
        }

        __syncthreads();  // Ensure all threads finish using shared memory
    }

    // Store result in output (ensure within bounds)
    if (row < seq_len && col < d_k) {
        output[batch_idx * seq_len * d_k + row * d_k + col] = sum;
    }

}

torch::Tensor attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int batch_size = Q.size(0);
    const int seq_len = Q.size(1);
    const int d_k = Q.size(2);

    auto QK_product = torch::zeros_like(Q);
    auto output = torch::zeros_like(Q);

    // Launch the CUDA kernel
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((seq_len+BLOCK_SIZE-1)/BLOCK_SIZE, (d_k+BLOCK_SIZE-1)/BLOCK_SIZE, batch_size);
    dim3 blocks2((seq_len+BLOCK_SIZE-1)/BLOCK_SIZE, (seq_len+BLOCK_SIZE-1)/BLOCK_SIZE, batch_size);

    key_query_multiplication<<<blocks2, threads>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), QK_product.data_ptr<float>(), batch_size, seq_len, d_k
    );

    scaled_dot_product_attention<<<blocks, threads>>>(
        QK_product.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(), batch_size, seq_len, d_k
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "CUDA Scaled Dot-Product Attention");
}
