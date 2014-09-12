#include "sentiment_kernels.h"

#include <ctime>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_primitives.h"

/************************/
/***** CUDA Kernels *****/
/************************/

__global__ void
setupRandomVectorGen(curandState* state, unsigned long seed, unsigned int numElems) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    curand_init(seed, id, 0, &(state[id]));
}

__global__ void
runRandomVectorGen(float* vec, curandState* globalState, float threshold, unsigned int numElems) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    curandState localState = globalState[id];
    float rndVal = curand_uniform(&localState);
    // scale to (-threshold, +threshold)
    vec[id] = (rndVal * 2 * threshold) - threshold;
}

__global__ void
updateParams(float* params, float* derivatives, float* weights, float learningRate, unsigned int numElems) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems) return;
    float epsilon = 0.0001;
    weights[id] += derivatives[id]*derivatives[id];
    params[id] -= (learningRate * derivatives[id])/(sqrt(weights[id]) + epsilon);
}

/***************************/
/***** Kernel Wrappers *****/
/***************************/

void
kernelRandomWordVectors(cParamMem_t& params, float threshold) {
    timeval tim;
    gettimeofday(&tim, NULL);
    double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
    
    // cudaEvent_t sync_event;
    // checkCudaErrors(cudaEventCreate(&sync_event));

    unsigned int blockSize = 1024;
    unsigned int numElems = params.numWords * params.wordDim;
    unsigned int numBlocks = numElems / blockSize + 1;
    dim3 threadsPerBlock(blockSize, 1, 1);
    curandState* devState;
    checkCudaErrors(cudaMalloc((void**)&devState, numElems*sizeof(curandState)));
    setupRandomVectorGen<<<numBlocks, threadsPerBlock>>>(devState, time(NULL), numElems);
    checkCudaErrors(cudaGetLastError());
    runRandomVectorGen<<<numBlocks, threadsPerBlock>>>(params.wordVectors, devState, threshold, numElems);
    checkCudaErrors(cudaThreadSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(devState));

    // checkCudaErrors(cudaEventRecord(sync_event));
    // checkCudaErrors(cudaEventSynchronize(sync_event));

    gettimeofday(&tim, NULL);
    double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    printf("Random word vectors time: %f\n", t2-t1);
}

void
kernelUpdateParams(cParamMem_t& params, cParamMem_t& derivatives,
  cParamMem_t& adagradWts, float learningRate) {
    timeval tim;
    gettimeofday(&tim, NULL);
    double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

    // cudaEvent_t sync_event;
    // checkCudaErrors(cudaEventCreate(&sync_event));

    unsigned int blockSize = 1024;
    unsigned int numElems = params.totalSize;
    unsigned int numBlocks = numElems / blockSize + 1;
    dim3 threadsPerBlock(blockSize, 1, 1);
    updateParams<<<numBlocks,threadsPerBlock>>>(params.base, derivatives.base,
        adagradWts.base, learningRate, params.totalSize);
    checkCudaErrors(cudaThreadSynchronize());
    checkCudaErrors(cudaGetLastError());

    // checkCudaErrors(cudaEventRecord(sync_event));
    // checkCudaErrors(cudaEventSynchronize(sync_event));

    gettimeofday(&tim, NULL);
    double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
    printf("Update params time: %f\n", t2-t1);
}
