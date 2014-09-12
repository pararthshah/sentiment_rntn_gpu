#include "model.h"

SentimentModel::SentimentModel() {
    mWordDim = -1;
    mNumClasses = -1;
    mNumWords = -1;
}

SentimentModel::SentimentModel(int wordDim, int numClasses, std::vector<Tree*>& trainTrees) {
    mWordDim = wordDim;
    mNumClasses = numClasses;
    initWordIdMap(trainTrees);

    // allocate and initialize parameter matrices
    // mSoftmaxW = CudaInterface::allocMirrorMem(mNumClasses*(mWordDim+1)); // C x (D+1)
    // mTransformW = CudaInterface::allocMirrorMem(mWordDim*(2*mWordDim+1)); // D x (2D+1)
    // for (int i = 0; i < mWordDim; i++) {
    //     mTransformV.push_back(CudaInterface::allocMirrorMem(4*mWordDim*mWordDim)); // 2D x 2D x D
    // }
    // mWordVectors = CudaInterface::allocMirrorMem(mWordDim*mNumWords); // D x L

    CudaInterface::allocParamMem(mParams_d, mWordDim, mNumClasses, mNumWords, true);
    CudaInterface::allocParamMem(mParams_h, mWordDim, mNumClasses, mNumWords, false);
}

SentimentModel::~SentimentModel() {
    CudaInterface::freeParamMem(mParams_d);
    CudaInterface::freeParamMem(mParams_h);
}

void
SentimentModel::initWordIdMap(std::vector<Tree*>& trainTrees) {
    std::set<std::string> words;
    for (auto tree: trainTrees) {
        SentimentUtils::getLeafWords(tree, words);
    }
    words.insert(UNSEEN_WORD);
    int wordId = 0;
    for (auto word: words) {
        mWordToId[word] = wordId;
        wordId++;
    }
    mNumWords = wordId;
    printf("num train trees: %lu, num words: %d\n", trainTrees.size(), mNumWords);
    for (auto tree: trainTrees) {
        SentimentUtils::assignLeafWordIds(tree, mWordToId, mWordToId[UNSEEN_WORD]);
    }
}
