#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <map>
#include <vector>

#include "cuda_interface.h"

#include "tree.h"
#include "utils.h"

class SentimentModel {
  public:
    int mWordDim; // D
    int mNumClasses; // C
    int mNumWords; // L

    // cMirrorMem_t mSoftmaxW; // C x (D+1)
    // cMirrorMem_t mTransformW; // D x (2D+1)
    // std::vector<cMirrorMem_t> mTransformV; // 2D x 2D x D
    // cMirrorMem_t mWordVectors; // D x L

    cParamMem_t mParams_h; // parameters in host memory
    cParamMem_t mParams_d; // parameters in device memory

    std::map<std::string, int> mWordToId;

    std::string UNSEEN_WORD = "<unseen>";

    SentimentModel();
    SentimentModel(int wordDim, int numClasses, std::vector<Tree*>& trainTrees);
    ~SentimentModel();

    void initWordIdMap(std::vector<Tree*>& trainTrees);
};

#endif // MODEL_H
