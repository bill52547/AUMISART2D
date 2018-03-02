// header for Alternately Updating Motion and Image SART method
#ifndef _AUMISART_H
#define _AUMISART_H
#include "universal.h"

#include "cu_add.h"
#include "cu_initial.h"
#include "CG_A_operation.h"
#include "cu_vector_multiply.h"
__host__ void host_CG(float *h_outimg, float *h_img, float *h_b, int nx, int ny, int na, int n_views, int n_iter_inner, float da, float ai, float SO, float SD, float mu, float *volumes, float *flows, float *angles, float *alpha_x, float *alpha_y, float *beta_x, float *beta_y);
#endif // _AUMISART_H