#ifndef TRAINING_H
#define TRAINING_H

#include <vector>
#include <string>

#include "cuda_interface.h"

#include "tree.h"

typedef struct _trainingOptions {
  std::string trainPath;
  std::string devPath;
  int batchSize;
  int numCycles;
  float learningRate;
  float regSoftmaxW;
  float regTransformW;
  float regTransformV;
  float regWordVectors;
} sOptions_t;

typedef struct _sentimentModel {
    int wordDim; // D
    int numClasses; // C
    int numWords; // L
    std::map<std::string, int> wordToId;

    cParamMem_t params_h; // parameters in host memory
    cParamMem_t params_d; // parameters in device memory
    cParamMem_t derivatives_d; // derivatives in device memory
    cParamMem_t adagradWts_d; // weights in device memory

    float* nodeClassDist_d; // temp node class distribution in device memory
    float* nodeVector_d;    // temp node vector in device memory
} sModel_t;

class SentimentTraining {
  public:
    SentimentTraining(sOptions_t options, int wordDim, int numClasses);
    void train();

  private:
    void initNodeVectors();
    void trainBatch(int startIndex, int endIndex);
    float computeDerivatives(int startIndex, int endIndex);
    void forwardPropagate(Tree* tree);
    void backPropagate(Tree* tree);

    std::vector<Tree*> mTrainTrees;
    std::vector<Tree*> mDevTrees;
    sModel_t mModel;
    sOptions_t mOptions;

    static const std::string UNSEEN_WORD;
};

#endif // TRAINING_H
