#include <iostream>

__global__ void arrayMultKernel(float *a, float *b, float *c, int n)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if(Col < n && Row < n){

    }
}

int main()
{
    // Initialize and allocate memory TODO


    return 0;
}