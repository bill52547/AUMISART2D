#ifndef _CU_DOT_PRODUCT_H
#define _CU_DOT_PRODUCT_H
#include "universal.h"
__host__ void host_dot(float *c, float *a, float *b, int nx, int ny);
__global__ void dot(float* c, float* a, float* b, int N);



#endif // _CU_DOT_PRODUCT_H