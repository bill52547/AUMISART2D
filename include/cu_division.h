#ifndef _CU_DIVISION_H
#define _CU_DIVISION_H
#include "universal.h"

__host__ void host_division(float *img1, float *img, int nx, int ny);
__global__ void kernel_division(float *img1, float *img, int nx, int ny);
#endif // _CU_DIVISION_H