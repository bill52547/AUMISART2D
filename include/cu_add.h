#ifndef _CU_ADD_h
#define _CU_ADD_h
#include "universal.h"
__host__ void host_add(float *img1, float *img, int nx, int ny, float weight);

__global__ void kernel_add(float *img1, float *img, int nx, int ny, float weight);

__host__ void host_add2(float *img1, float *img, int nx, int ny, float* img0, float weight, int ind);

__global__ void kernel_add2(float *img1, float *img, int nx, int ny, float* img0, float weight, int ind);
#endif // _CU_ADD_h
