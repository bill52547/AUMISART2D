__global__ void kernel_invert(float *mx2, float *my2, float *mz2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, cudaTextureObject_t tex_mz, int nx, int ny, int nz);
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define MX prhs[0]
#define MY prhs[1]
#define MZ prhs[2]
#define PARA prhs[3]

#define OUT_MX plhs[0]
#define OUT_MY plhs[1]
#define OUT_MZ plhs[2]

float *h_mx, *h_my, *h_mz;
h_mx = (float*)mxGetData(MX);
h_my = (float*)mxGetData(MY);
h_mz = (float*)mxGetData(MZ);
int nx, ny, nz;
nx = (int)mxGetScalar(mxGetField(PARA, 0, "nx"));
ny = (int)mxGetScalar(mxGetField(PARA, 0, "ny"));
nz = (int)mxGetScalar(mxGetField(PARA, 0, "nz"));

OUT_MX = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
OUT_MY = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
OUT_MZ = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
mwSize outDim[3] = {(mwSize)nx, (mwSize)ny, (mwSize)nz};
mxSetDimensions(OUT_MX, outDim, 3);
mxSetDimensions(OUT_MY, outDim, 3);
mxSetDimensions(OUT_MZ, outDim, 3);
mxSetData(OUT_MX, mxMalloc(nx * ny * nz * sizeof(float)));
mxSetData(OUT_MY, mxMalloc(nx * ny * nz * sizeof(float)));
mxSetData(OUT_MZ, mxMalloc(nx * ny * nz * sizeof(float)));


float *h_mx2 = (float*)mxGetData(OUT_MX);
float *h_my2 = (float*)mxGetData(OUT_MY);
float *h_mz2 = (float*)mxGetData(OUT_MZ);

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaPitchedPtr dp_mx = make_cudaPitchedPtr((void*) h_mx, nx * sizeof(float), nx, ny);
cudaPitchedPtr dp_my = make_cudaPitchedPtr((void*) h_my, nx * sizeof(float), nx, ny);
cudaPitchedPtr dp_mz = make_cudaPitchedPtr((void*) h_mz, nx * sizeof(float), nx, ny);

cudaMemcpy3DParms copyParams = {0};
struct cudaExtent extent = make_cudaExtent(nx, ny, nz);
copyParams.extent = extent;
copyParams.kind = cudaMemcpyHostToDevice;

copyParams.srcPtr = dp_mx;
cudaArray *array_mx;
cudaMalloc3DArray(&array_mx, &channelDesc, extent);
copyParams.dstArray = array_mx;
cudaMemcpy3D(&copyParams);   

copyParams.srcPtr = dp_my;
cudaArray *array_my;
cudaMalloc3DArray(&array_my, &channelDesc, extent);
copyParams.dstArray = array_my;
cudaMemcpy3D(&copyParams);   

copyParams.srcPtr = dp_mz;
cudaArray *array_mz;
cudaMalloc3DArray(&array_mz, &channelDesc, extent);
copyParams.dstArray = array_mz;
cudaMemcpy3D(&copyParams);   


cudaResourceDesc resDesc;
cudaTextureDesc texDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeClamp;
texDesc.addressMode[1] = cudaAddressModeClamp;
texDesc.addressMode[2] = cudaAddressModeClamp;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 0;

resDesc.res.array.array = array_mx;
cudaTextureObject_t tex_mx = 0;
cudaCreateTextureObject(&tex_mx, &resDesc, &texDesc, NULL);

resDesc.res.array.array = array_my;
cudaTextureObject_t tex_my = 0;
cudaCreateTextureObject(&tex_my, &resDesc, &texDesc, NULL);

resDesc.res.array.array = array_mz;
cudaTextureObject_t tex_mz = 0;
cudaCreateTextureObject(&tex_mz, &resDesc, &texDesc, NULL);
const dim3 gridSize((nx + 16 - 1) / 16, (ny + 16 - 1) / 16, (nz + 4 - 1) / 4);
const dim3 blockSize(16, 16, 4);

float *d_mx2, *d_my2, *d_mz2;
cudaMalloc((void**)&d_mx2, nx * ny * nz * sizeof(float));
cudaMalloc((void**)&d_my2, nx * ny * nz * sizeof(float));
cudaMalloc((void**)&d_mz2, nx * ny * nz * sizeof(float));

kernel_invert<<<gridSize, blockSize>>>(d_mx2, d_my2, d_mz2, tex_mx, tex_my, tex_mz, nx, ny, nz);
cudaDeviceSynchronize();
cudaDestroyTextureObject(tex_mx);
cudaFreeArray(array_mx);
cudaDestroyTextureObject(tex_my);
cudaFreeArray(array_my);
cudaDestroyTextureObject(tex_mz);
cudaFreeArray(array_mz);


cudaMemcpy(h_mx2, d_mx2, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(h_my2, d_my2, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(h_mz2, d_mz2, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(d_mx2);
cudaFree(d_my2);
cudaFree(d_mz2);

cudaDeviceReset();
return;
}
__global__ void kernel_invert(float *mx2, float *my2, float *mz2, cudaTextureObject_t tex_mx, cudaTextureObject_t tex_my, cudaTextureObject_t tex_mz, int nx, int ny, int nz)
{
    int ix = 16 * blockIdx.x + threadIdx.x;
    int iy = 16 * blockIdx.y + threadIdx.y;
    int iz = 4 * blockIdx.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz)
        return;
    int id = ix + iy * nx + iz * nx * ny;
    float x = 0, y = 0, z = 0;
    for (int iter = 0; iter < 10; iter ++){
        x = - tex3D<float>(tex_mx, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
        y = - tex3D<float>(tex_my, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
        z = - tex3D<float>(tex_mz, (x + ix + 0.5f), (y + iy + 0.5f), (z + iz + 0.5f));
    }
    mx2[id] = x;
    my2[id] = y;
    mz2[id] = z;
}