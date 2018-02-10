#ifndef _CU_DIFF_H
#define _CU_DIFF_H
#include "universal.h"
__device__ float diff_x(float *img, int ix, int iy, int nx, int ny);
__device__ float diff_y(float *img, int ix, int iy, int nx, int ny);
#endif // _CU_DIFF_H