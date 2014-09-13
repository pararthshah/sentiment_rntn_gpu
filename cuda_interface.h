#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdlib>

#include "cuda_primitives.h"
#include "sentiment_kernels.h"

class CudaInterface {
  public:
    static void initialize();
    static void cleanup();

    /***** Host/Device Memory Management *****/
    static void allocMem(float** mem, unsigned int size, bool device);
    static void freeMem(float* mem, bool device);
    static void transferMem(cParamMem_t pmem1, cParamMem_t pmem2, bool device1, bool device2);

    static void allocParamMem(cParamMem_t& pmem, unsigned wordDim, unsigned numClasses, unsigned numWords, bool device);
    static void freeParamMem(cParamMem_t& pmem);
    static void fillParamMem(cParamMem_t& pmem, int byteVal);

    /***** CUBLAS wrappers *****/
    static int cublasMatrixMult(float* A, float* B, float* C, cMatrixSize_t mSize);

    /***** Unused *****/
    CudaInterface() {}
 
  private:
    static const float mAlpha;
    static const float mBeta;
    static int mDevID;
    static cudaDeviceProp mDeviceProp;
    static cublasHandle_t mCublasHandle;
};

#endif // CUDA_INTERFACE_H
