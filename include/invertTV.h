#ifndef _INVERT_TV_H
#define _INVERT_TV_H
#include "universal.h"
__host__ void host_wtx(float* d_x, float *d_wx, int nx, int ny);
__global__ void kernel_wtx(float *x, float *wx, int nx, int ny);
#endif // _INVERT_TV_H