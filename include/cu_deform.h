#ifndef _CU_DEFORM_H
#define _CU_DEFORM_H
#include "universal.h"

__host__ void host_deform(float *d_img1, float *d_img, int nx, int ny, float volume_diff, float flow_diff, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);
__host__ void host_deform2(float *d_img1, float *d_img, int nx, int ny, float volume_diff, float flow_diff, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);
__host__ void host_deform_invert(float *d_img1, float *d_img, int nx, int ny, float volume, float flow, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);

__global__ void kernel_forwardDVF(float *mx, float *my, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y, float volume, float flow, int nx, int ny);
__global__ void kernel_deformation(float *img1, cudaTextureObject_t tex_img, float *mx2, float *my2, int nx, int ny);
__global__ void kernel_deformation2(float *img1, float *img, float *mx, float *my, int nx, int ny);
__host__ void host_invert(float *mx2, float *my2, float *mx, float *my, int nx, int ny);
__global__ void kernel_invert(float *mx2, float *my2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, int nx, int ny);
#endif // _CU_DEFORM_H