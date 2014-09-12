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

float*
CudaInterface::allocMem(unsigned int size) {
    float* mem;
    checkCudaErrors(cudaMalloc((void **) &mem, size * sizeof(float)));
    return mem;
}

void
CudaInterface::freeMem(float* mem) {
    checkCudaErrors(cudaFree(mem));
}

cMirrorMem_t
CudaInterface::allocMirrorMem(unsigned int size) {
    cMirrorMem_t mmem;
    unsigned int mem_size = size * sizeof(float);
    mmem.host = (float *) malloc(mem_size);
    checkCudaErrors(cudaMalloc((void **) &(mmem.device), mem_size));
    mmem.size = size;
    return mmem;
}

void
CudaInterface::freeMirrorMem(cMirrorMem_t mmem, unsigned int flag) {
    if (flag == 1 || flag == 3)
        free(mmem.host);
    if (flag == 2 || flag == 3) 
        checkCudaErrors(cudaFree(mmem.device));
}

void
CudaInterface::copyHostToDevice(cMirrorMem_t mmem) {
    checkCudaErrors(cudaMemcpy(mmem.device, mmem.host, mmem.size * sizeof(float), cudaMemcpyHostToDevice));
}

void
CudaInterface::copyDeviceToHost(cMirrorMem_t mmem) {
    checkCudaErrors(cudaMemcpy(mmem.host, mmem.device, mmem.size * sizeof(float), cudaMemcpyDeviceToHost));
}

void
CudaInterface::allocParamMem(cParamMem_t& pmem, unsigned wordDim, unsigned numClasses, unsigned numWords, bool device) {
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
        // checkCudaErrors(cudaEventCreate(&sync_event));
        checkCudaErrors(cudaMalloc((void **) &(pmem.base), pmem.totalSize * sizeof(float)));
        checkCudaErrors(cudaThreadSynchronize());
        // checkCudaErrors(cudaEventRecord(sync_event));
        // checkCudaErrors(cudaEventSynchronize(sync_event));
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
    if (pmem.device)
        checkCudaErrors(cudaFree(pmem.base));
    else
        free(pmem.base);
    pmem.base = nullptr;
    pmem.softmaxW = nullptr;
    pmem.transformW = nullptr;
    pmem.transformV = nullptr;
    pmem.wordVectors = nullptr;
}

void
CudaInterface::fillParamMem(cParamMem_t& pmem, int byteVal) {
    printf("setting %ld bytes at %x\n", pmem.totalSize * sizeof(float), pmem.base);
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
    checkCudaErrors(cublasSgemm(mCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, mSize.uiWB, mSize.uiHA, 
                                mSize.uiWA, &mAlpha, B, mSize.uiWB, A, mSize.uiWA, &mBeta, C, mSize.uiWA));
    return 0;
}
