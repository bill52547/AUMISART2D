#include "host_projection.h" // consists all required package and functions

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
// Macro for input and output
#define IN_IMG prhs[0]
#define GEO_PARA prhs[1]
#define OUT_PROJ plhs[0]

int nx, ny, nz, na, nb, numImg, numBytesImg, numSingleProj, numBytesSingleProj;
float da, db, ai, bi, SO, SD, angle;

// resolutions of volumes 
if (mxGetField(GEO_PARA, 0, "nx") != NULL)
    nx = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nx"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nx.\n");

if (mxGetField(GEO_PARA, 0, "ny") != NULL)
    ny = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "ny"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution ny.\n");

if (mxGetField(GEO_PARA, 0, "nz") != NULL)
    nz = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nz"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid volume resolution nz.\n");

numImg = nx * ny * nz; // size of image
numBytesImg = numImg * sizeof(float); // number of bytes in image

// detector plane resolutions
if (mxGetField(GEO_PARA, 0, "na") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "na"));
else if (mxGetField(GEO_PARA, 0, "nv") != NULL)
    na = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nv"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector in plane, which is denoted as na or nu.\n");

if (mxGetField(GEO_PARA, 0, "nb") != NULL)
    nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nb"));
else if (mxGetField(GEO_PARA, 0, "nu") != NULL)
    nb = (int)mxGetScalar(mxGetField(GEO_PARA, 0, "nu"));
else
	mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid number of detector across plane, which is denoted as nb or nv.\n");

numSingleProj = na * nb;
numBytesSingleProj = numSingleProj * sizeof(float);

// detector resolution
if (mxGetField(GEO_PARA, 0, "da") != NULL)
    da = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "da"));
else{
    da = 1.0f;
    mexPrintf("Automatically set detector cell size da to 1. \n");
    mexPrintf("If don't want that default value, please set para.da manually.\n");
}

if (mxGetField(GEO_PARA, 0, "db") != NULL)
    db = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "db"));
else{
    db = 1.0f;
    mexPrintf("Automatically set detectof cell size db to 1. \n");
    mexPrintf("If don't want that default value, please set para.db manually.\n");
}


// detector plane offset from centered calibrations
if (mxGetField(GEO_PARA, 0, "ai") != NULL){
    ai = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "ai"));
    ai -= (float)na / 2 - 0.5f;
}
else{
    mexPrintf("Automatically set detector offset ai to 0. \n");
    mexPrintf("If don't want that default value, please set para.ai manually.\n");
    ai = - (float)na / 2 + 0.5f;
}

if (mxGetField(GEO_PARA, 0, "bi") != NULL){
    bi = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "bi"));
    if (bi > -1)
        bi -= (float)nb / 2 - 0.5f;
}
else{
    mexPrintf("Automatically set detector offset bi to 0. \n");
    mexPrintf("If don't want that default value, please set para.bi manually.\n");
    bi = - (float)nb / 2 + 0.5f;
}


if (mxGetField(GEO_PARA, 0, "SO") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SO"));
else if (mxGetField(GEO_PARA, 0, "SI") != NULL)
    SO = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SI"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and isocenter, which is denoted with para.SO or para.DI.\n");

if (mxGetField(GEO_PARA, 0, "SD") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "SD"));
else if (mxGetField(GEO_PARA, 0, "DI") != NULL)
    SD = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "DI")) + SO;
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid distance between source and detector plane, which is denoted with para.SD or para.SI + para.DI.\n");

if (mxGetField(GEO_PARA, 0, "angle") != NULL)
    angle = (float)mxGetScalar(mxGetField(GEO_PARA, 0, "angle"));
else
    mexErrMsgIdAndTxt("MATLAB:badInput","Can't found valid projection angle, which is denoted with para.angle.\n");

float *d_img, *d_proj;
cudaMalloc((void**)&d_img, nx * ny * nz * sizeof(float));

float *h_img;
h_img = (float*)mxGetData(IN_IMG);
cudaMemcpy(d_img, h_img, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void**)&d_proj, na * nb * sizeof(float));
const dim3 gridSize_singleProj((na + BLOCKSIZE_X - 1) / BLOCKSIZE_X, (nb + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y, 1);
const dim3 blockSize(BLOCKSIZE_X,BLOCKSIZE_Y, 1);

kernel_projection<<<gridSize_singleProj, blockSize>>>(d_proj, d_img, angle, SO, SD, da, na, ai, db, nb, bi, nx, ny, nz);
cudaDeviceSynchronize();

OUT_PROJ = mxCreateNumericMatrix(0, 0, mxSINGLE_CLASS, mxREAL);
const mwSize outDim[2] = {(mwSize)na, (mwSize)nb};

mxSetDimensions(OUT_PROJ, outDim, 2);
mxSetData(OUT_PROJ, mxMalloc(na * nb * sizeof(float)));
float *h_outproj = (float*)mxGetData(OUT_PROJ);

cudaMemcpy(h_outproj, d_proj, numBytesSingleProj, cudaMemcpyDeviceToHost);

cudaFree(d_proj);

cudaFree(d_img);
cudaDeviceReset();
return;
}

