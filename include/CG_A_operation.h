#ifndef CG_A_OPERATION_H
#define CG_A_OPERATION_H
#include "universal.h"
#include "cu_add.h"
#include "cu_deform.h"
#include "cu_initial.h"
#include "cu_projection.h"
#include "cu_backprojection.h"
#include "forwardTV.h"
#include "invertTV.h"
#include "cu_vector_multiply.h"
__host__ void A_operation(float *d_xp, float *d_img, int nx, int ny, int na, int n_views, float da, float ai, float SO, float SD, float mu, float* volumes,float* flows, float* angles, float* d_alpha_x, float* d_alpha_y, float* d_beta_x, float* d_beta_y);

#endif // CG_A_OPERATION_H