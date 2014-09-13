#include "cuda_interface.h"

#include <cstring>

/*******************************/
/***** Interface Functions *****/
/*******************************/

const float CudaInterface::mAlpha = 1.0f;
const float CudaInterface::mBeta  = 0.0f;
int CudaInterface::mDevID;
cudaDeviceProp CudaInterface::mDeviceProp;
cublasHandle_t CudaInterface::mCublasHandle;

void
CudaInterface::initialize() {
    mDevID = 0;
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    checkCudaErrors(cudaGetDeviceProperties(&mDeviceProp, mDevID));
    checkCudaErrors(cublasCreate(&mCublasHandle));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", mDevID, mDeviceProp.name, mDeviceProp.major, mDeviceProp.minor);

    // use a larger block size for Fermi and above
    int block_size = (mDeviceProp.major < 2) ? 16 : 32;
}

void
CudaInterface::cleanup() {
    checkCudaErrors(cublasDestroy(mCublasHandle));
    //cudaDeviceReset();
}

/*****************************/
/***** Memory Management *****/
/*****************************/

void
CudaInterface::allocMem(float** mem, unsigned int size, bool device) {
    if (device)
        checkCudaErrors(cudaMalloc((void **) mem, size * sizeof(float)));
    else
        *mem = (float*) malloc(size * sizeof(float));
}

void
CudaInterface::freeMem(float* mem, bool device) {
    if (device)
        checkCudaErrors(cudaFree(mem));
    else
        free(mem);
}

void
CudaInterface::transferMem(cParamMem_t pmem1, cParamMem_t pmem2, bool device1, bool device2) {
    cudaMemcpyKind kind;
    if (device1 && device2) kind = cudaMemcpyDeviceToDevice;
    else if (!device1 && device2) kind = cudaMemcpyDeviceToHost;
    else if (device1 && !device2) kind = cudaMemcpyHostToDevice;
    else  kind = cudaMemcpyHostToHost;

    checkCudaErrors(cudaMemcpy(pmem1.base, pmem2.base, pmem1.totalSize * sizeof(float), kind));
}

void
CudaInterface::allocParamMem(cParamMem_t& pmem, unsigned wordDim, unsigned numClasses, unsigned numWords, bool device) {
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    pmem.wordDim = wordDim;
    pmem.numClasses = numClasses;
    pmem.numWords = numWords;
    pmem.device = device;
    pmem.totalSize  = numClasses * (wordDim+1);
    pmem.totalSize += wordDim * (2*wordDim+1);
    pmem.totalSize += 4 * wordDim * wordDim * wordDim;
    pmem.totalSize += numWords * wordDim; // 5*33+32*65+4*1024*32+16582*32
    if (device) {
        cudaEvent_t sync_event;
        checkCudaErrors(cudaMalloc((void **) &(pmem.base), pmem.totalSize * sizeof(float)));
    } else {
        pmem.base = (float*) malloc(pmem.totalSize * sizeof(float));
    }
    printf("allocated %lu bytes to %x\n", pmem.totalSize * sizeof(float), pmem.base);
    pmem.softmaxW = pmem.base;
    pmem.transformW = pmem.softmaxW + numClasses * (wordDim+1);
    pmem.transformV = pmem.transformW + numClasses * (2*wordDim+1);
    pmem.wordVectors = pmem.transformV + 4 * wordDim * wordDim * wordDim;
}

void
CudaInterface::freeParamMem(cParamMem_t& pmem) {
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    if (pmem.device)
        checkCudaErrors(cudaFree(pmem.base));
    else
        free(pmem.base);
    pmem.base = NULL;
    pmem.softmaxW = NULL;
    pmem.transformW = NULL;
    pmem.transformV = NULL;
    pmem.wordVectors = NULL;
}

void
CudaInterface::fillParamMem(cParamMem_t& pmem, int byteVal) {
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    printf("  setting %ld bytes at %x\n", pmem.totalSize * sizeof(float), pmem.base);
    if (pmem.device) {
        checkCudaErrors(cudaThreadSynchronize());
        checkCudaErrors(cudaMemset(pmem.base, byteVal, pmem.totalSize * sizeof(float)));
        checkCudaErrors(cudaThreadSynchronize());
    } else
        memset(pmem.base, byteVal, pmem.totalSize * sizeof(float));
}

/***************************/
/***** CUBLAS Wrappers *****/
/***************************/

int
CudaInterface::cublasMatrixMult(float* A, float* B, float* C, cMatrixSize_t mSize) {
    checkCudaErrors(cudaSetDevice(mDevID));
    checkCudaErrors(cudaGetDevice(&mDevID));
    checkCudaErrors(cublasSgemm(mCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, mSize.uiWB, mSize.uiHA, 
                                mSize.uiWA, &mAlpha, B, mSize.uiWB, A, mSize.uiWA, &mBeta, C, mSize.uiWA));
    return 0;
}
