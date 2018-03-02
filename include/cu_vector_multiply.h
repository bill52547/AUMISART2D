__global__ void kernel_multiply(float *c, float *a, float *b, int nx, int ny);

__host__ void cu_vector_multiply(float *c, float *a, float *b, int nx, int ny);