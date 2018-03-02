#ifndef _FORWARD_TV_H
#define _FORWARD_TV_H
#include "universal.h"
__host__ void host_wx(float* d_wx, float *d_x, int nx, int ny);
__global__ void kernel_wx(float *wx, float *x, int nx, int ny);
#endif // _FORWARD_TV_H