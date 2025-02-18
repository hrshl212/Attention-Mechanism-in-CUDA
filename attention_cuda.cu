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
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Sequence index 
    int col = blockIdx.x * blockDim.x + threadIdx.x; // d_k index 

    if (row < seq_len && col < seq_len){

        // Calculate the dot product for each head at each position
        // Load Q, K matrices into shared memory
        __shared__ float shared_Q[BLOCK_SIZE][BLOCK_SIZE];  // Using 32x32 shared memory for Q
        __shared__ float shared_K[BLOCK_SIZE][BLOCK_SIZE];  // Using 32x32 shared memory for K

        // Step 1: Compute dot product between Q and K
        float dot_product = 0.0f;
        for (int i = 0; i < (d_k + BLOCK_SIZE-1) / BLOCK_SIZE; i++) {
            if(i * BLOCK_SIZE + threadIdx.x < d_k)
                shared_Q[threadIdx.y][threadIdx.x] = Q[batch_idx * seq_len * d_k + row * d_k + (i * BLOCK_SIZE+threadIdx.x)];
            else
                shared_Q[threadIdx.y][threadIdx.x] = 0.0f;

            if(i * BLOCK_SIZE + threadIdx.y < d_k)
                shared_K[threadIdx.x][threadIdx.y] = K[batch_idx * seq_len * d_k + col * d_k + (i * BLOCK_SIZE+threadIdx.y)];
            else
                shared_K[threadIdx.x][threadIdx.y] = 0.0f;

            __syncthreads();

            for (int n = 0; n < BLOCK_SIZE; n++) {
                    dot_product += shared_Q[threadIdx.y][n] * shared_K[threadIdx.x][n];
            }        
            __syncthreads();
        }    

        // Scale the dot product by the square root of the key dimension
        dot_product /= sqrtf(d_k);
        QK_product[batch_idx * seq_len * seq_len + row * seq_len + col] = dot_product;
    }
}

__global__ void softmax_kernel(
    float* QK_product, 
    int seq_len) {
    
    int batch_idx = blockIdx.x;
    int row = threadIdx.x;
    // Compute max value in row for numerical stability
    float max_val = -1e20;
    for (int j = 0; j < seq_len; j++) {
        max_val = fmaxf(max_val, QK_product[batch_idx * seq_len * seq_len + row * seq_len + j]);
    }

    // Compute exponentials and sum
    float exp_sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        QK_product[batch_idx * seq_len * seq_len + row * seq_len + j] = expf(QK_product[batch_idx * seq_len * seq_len + row * seq_len + j] - max_val);
        exp_sum += QK_product[batch_idx * seq_len * seq_len + row * seq_len + j];
    }

    // Normalize by sum
    for (int j = 0; j < seq_len; j++) {
        QK_product[batch_idx * seq_len * seq_len + row * seq_len + j] /= exp_sum;
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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;  // Accumulate the result

    // Loop over tiles of QK and V
    for (int t = 0; t < (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load A and B tiles into shared memory (ensure within bounds)
        if (row < seq_len && (t * BLOCK_SIZE + threadIdx.x) < seq_len)
            QK_tile[threadIdx.y][threadIdx.x] = QK_product[batch_idx * seq_len * seq_len + row * seq_len + t * BLOCK_SIZE + threadIdx.x];
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

    auto QK_product = torch::zeros((batch_size, seq_len, seq_len),torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto output = torch::zeros_like(Q);

    // Launch the CUDA kernel
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((seq_len+BLOCK_SIZE-1)/BLOCK_SIZE, (d_k+BLOCK_SIZE-1)/BLOCK_SIZE, batch_size);
    dim3 blocks2((seq_len+BLOCK_SIZE-1)/BLOCK_SIZE, (seq_len+BLOCK_SIZE-1)/BLOCK_SIZE, batch_size);

    key_query_multiplication<<<blocks2, threads>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), QK_product.data_ptr<float>(), batch_size, seq_len, d_k
    );

    softmax_kernel<<<batch_size, seq_len>>>(
        QK_product.data_ptr<float>(), seq_len
    );

    scaled_dot_product_attention<<<blocks, threads>>>(
        QK_product.data_ptr<float>(), V.data_ptr<float>(), output.data_ptr<float>(), batch_size, seq_len, d_k
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "CUDA Scaled Dot-Product Attention");
}
