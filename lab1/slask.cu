#include <iostream>

const int block_size = 128;

__global__ void Add(float *X, float *Y, float ans, long N)
{
    __shared__ float buffer[block_size];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    buffer[tid] = X[i] * Y[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            buffer[tid] += buffer[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    atomicAdd(ans, buffer[0]); //TODO
}

int main(int argc, char *argv[])
{
    long N = std::stoll(argv[1]);

    float *h_A, *h_B, *d_A, *d_B;
    float h_ans = 0.f, d_ans = 0.f;

    // host vectors
    h_A = (float *)malloc(N * sizeof(float));
    h_B = (float *)malloc(N * sizeof(float));

    // device vectors.
    cudaMalloc((void **)&d_A, N * sizeof(float));
    cudaMalloc((void **)&d_B, N * sizeof(float));

    // Create two random vectors.
    srand((unsigned)time(NULL));
    float sumQuad = 0.f, a = 0.f, b = 0.f;
    for (int i = 0; i < N; i++)
    {
        a = (float)rand() / RAND_MAX;
        b = (float)rand() / RAND_MAX;
        h_A[i] = a;
        h_B[i] = b;
        sumQuad += a * b;
    }

    // Copy Host vrctors to device
    cudaMemcpy(d_A, h_A, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N, cudaMemcpyHostToDevice);

    // Call kernel
    cudaMemset(d_ans, 0, sizeof(float)); //TODO
    int grid_size = (N + block_size - 1) / block_size;
    Add<<<grid_size, block_size>>>(d_A, d_B, d_ans, N);
    cudaMemcpy(&h_ans, &d_ans, 1, cudaMemcpyDeviceToHost);

    // Print result.
    printf("Serial : %f\n", sumQuad);
    printf("CUDA: %f\n", h_ans);
}
