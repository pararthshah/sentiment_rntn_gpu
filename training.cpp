#include "training.h"

#include <algorithm>
#include <unistd.h>

#include "sentiment_kernels.h"

const std::string SentimentTraining::UNSEEN_WORD = "<unseen>";

SentimentTraining::SentimentTraining(sOptions_t options, int wordDim, int numClasses) {
    mOptions = options;
    mModel.wordDim = wordDim;
    mModel.numClasses = numClasses;
}

void
SentimentTraining::initNodeVectors() {
    std::set<std::string> words;
    Tree::getAllLeafWords(mTrainTrees, words);
    words.insert(UNSEEN_WORD);
    int wordId = 0;
    std::set<std::string>::iterator it;
    for (it = words.begin(); it != words.end(); ++it) {
        mModel.wordToId[*it] = wordId;
        wordId++;
    }
    mModel.numWords = wordId;
    printf("num train trees: %lu, num words: %d\n", mTrainTrees.size(), mModel.numWords);
    Tree::assignAllNodeVectorsAndId(mTrainTrees, mModel.wordToId, mModel.wordToId[UNSEEN_WORD],
      mModel.wordDim, mModel.numClasses);
}

void
SentimentTraining::train() {
    CudaInterface::initialize();
    Tree::readTrees(mTrainTrees, mOptions.trainPath);
    Tree::readTrees(mDevTrees, mOptions.devPath);

    int numBatches = mTrainTrees.size() / mOptions.batchSize + 1;

    initNodeVectors();

    CudaInterface::allocParamMem(mModel.params_d, mModel.wordDim,
        mModel.numClasses, mModel.numWords, true);

    CudaInterface::allocParamMem(mModel.params_h, mModel.wordDim,
        mModel.numClasses, mModel.numWords, false);

    CudaInterface::allocParamMem(mModel.derivatives_d, mModel.wordDim,
        mModel.numClasses, mModel.numWords, true);

    CudaInterface::allocParamMem(mModel.adagradWts_d, mModel.wordDim,
        mModel.numClasses, mModel.numWords, true);

    CudaInterface::allocMem(&(mModel.nodeClassDist_d), mModel.numClasses, true);
    CudaInterface::allocMem(&(mModel.nodeVector_d), mModel.wordDim, true);

    CudaInterface::fillParamMem(mModel.params_d, 0);
    kernelRandomWordVectors(mModel.params_d, 0.001);

    for (unsigned cycle = 0; cycle < mOptions.numCycles; cycle++) {
        std::random_shuffle(mTrainTrees.begin(), mTrainTrees.end());

        CudaInterface::fillParamMem(mModel.adagradWts_d, 0);

        for (unsigned batch = 0; batch < numBatches; batch++) {
            int startIndex =     batch * mOptions.batchSize;
            int endIndex   = (batch+1) * mOptions.batchSize;
            if (startIndex >= mTrainTrees.size()) break;
            if (endIndex + mOptions.batchSize > mTrainTrees.size()) endIndex = mTrainTrees.size();

            printf("Cycle: %u, Batch: %u\n", cycle, batch);
            trainBatch(startIndex, endIndex);
        }
    }

    CudaInterface::freeParamMem(mModel.params_d);
    CudaInterface::freeParamMem(mModel.params_h);
    CudaInterface::freeParamMem(mModel.derivatives_d);
    CudaInterface::freeParamMem(mModel.adagradWts_d);
    CudaInterface::freeMem(mModel.nodeClassDist_d, true);
    CudaInterface::freeMem(mModel.nodeVector_d, true);
    Tree::cleanupTrees(mTrainTrees);
    Tree::cleanupTrees(mDevTrees);
    CudaInterface::cleanup();
}

void
SentimentTraining::trainBatch(int startIndex, int endIndex) {
    float value = computeDerivatives(startIndex, endIndex);
    kernelUpdateParams(mModel.params_d, mModel.derivatives_d, mModel.adagradWts_d, mOptions.learningRate);
}

float
SentimentTraining::computeDerivatives(int startIndex, int endIndex) {
    CudaInterface::fillParamMem(mModel.derivatives_d, 0);

    for (int i = startIndex; i < endIndex; i++) {
        forwardPropagate(mTrainTrees[i]);
    }

    for (int i = startIndex; i < endIndex; i++) {
        backPropagate(mTrainTrees[i]);
    }

    return 1.0;
}

void
SentimentTraining::forwardPropagate(Tree* tree) {
    
}

void
SentimentTraining::backPropagate(Tree* tree) {
    
}
