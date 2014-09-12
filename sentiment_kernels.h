#ifndef SENTIMENT_KERNELS_H
#define SENTIMENT_KERNELS_H

#include <cstdlib>
#include <cstring>

typedef struct _matrixSize {
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} cMatrixSize_t;

typedef struct _mirrorMem {
    float* host;
    float* device;
    unsigned int size;
} cMirrorMem_t;

// encapsulates all the parameters of the model
// contiguous memory allocated on the device starts at paramBase
// all other pointers point directly to the start of respective matrices.
typedef struct _paramMem {
    float* base;
    float* softmaxW;
    float* transformW;
    float* transformV;
    float* wordVectors;
    unsigned int totalSize;
    unsigned int wordDim;
    unsigned int numClasses;
    unsigned int numWords;
    bool device;
} cParamMem_t;

/***** Kernel wrappers *****/
void kernelRandomWordVectors(cParamMem_t& params, float threshold);
void kernelUpdateParams(cParamMem_t& params, cParamMem_t& derivatives,
  cParamMem_t& adagradWts, float learningRate);

#endif // SENTIMENT_KERNELS_H