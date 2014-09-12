#include "training.h"

#include <algorithm>

#include "sentiment_kernels.h"

SentimentTraining::SentimentTraining(const std::string& trainPath, const std::string& devPath,
  int wordDim, int numClasses) {
    mTrainTrees = SentimentUtils::readTrees(trainPath);
    mDevTrees = SentimentUtils::readTrees(devPath);
    CudaInterface::initialize();
    mModel = SentimentModel(wordDim, numClasses, mTrainTrees);
}

SentimentTraining::~SentimentTraining() {
    CudaInterface::cleanup();
    SentimentUtils::cleanupTrees(mTrainTrees);
    SentimentUtils::cleanupTrees(mDevTrees);
}

void
SentimentTraining::train(sTrainingOptions_t options) {
    int numBatches = mTrainTrees.size() / options.batchSize + 1;
    cParamMem_t derivatives, adagradWts;

    CudaInterface::allocParamMem(derivatives, mModel.mWordDim,
        mModel.mNumClasses, mModel.mNumWords, true);

    CudaInterface::allocParamMem(adagradWts, mModel.mWordDim,
        mModel.mNumClasses, mModel.mNumWords, true);

    CudaInterface::fillParamMem(mModel.mParams_d, 0);    
    kernelRandomWordVectors(mModel.mParams_d, 0.001);

    for (unsigned cycle = 0; cycle < options.numCycles; cycle++) {
        std::random_shuffle(mTrainTrees.begin(), mTrainTrees.end());

        CudaInterface::fillParamMem(adagradWts, 0);

        for (unsigned batch = 0; batch < numBatches; batch++) {
            int startIndex =     batch * options.batchSize;
            int endIndex   = (batch+1) * options.batchSize;
            if (startIndex >= mTrainTrees.size()) break;
            if (endIndex + options.batchSize > mTrainTrees.size()) endIndex = mTrainTrees.size();

            printf("Cycle: %u, Batch: %u\n", cycle, batch);
            trainBatch(startIndex, endIndex, options, derivatives, adagradWts);
        }
    }    
}

void
SentimentTraining::trainBatch(int startIndex, int endIndex, const sTrainingOptions_t& options,
  cParamMem_t& derivatives, cParamMem_t& adagradWts) {
    
    float value = 0;
    computeDerivatives(startIndex, endIndex, options, derivatives, value);
    kernelUpdateParams(mModel.mParams_d, derivatives, adagradWts, options.learningRate);
}

void
SentimentTraining::computeDerivatives(int startIndex, int endIndex, const sTrainingOptions_t& options,
  cParamMem_t& derivatives, float& value) {
    CudaInterface::fillParamMem(derivatives, 0);
}

void
SentimentTraining::forwardPropagate(Tree* tree) {
    
}

void
SentimentTraining::backPropagate(Tree* tree) {
    
}
