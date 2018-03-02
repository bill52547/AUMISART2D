#include "cu_vector_multiply.h"

__host__ void cu_vector_multiply(float *c, float *a, float *b, int nx, int ny)
{
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    kernel_multiply<<<1,1>>>(c, a, b, nx, ny);
}

__global__ void kernel_multiply(float *c, float *a, float *b, int nx, int ny)
{
    *c = 0.0f;
    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < nx; iy++)
        {
            *c += a[ix + iy * nx] * b[ix + iy * nx];        }
    }
}