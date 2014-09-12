#ifndef TRAINING_H
#define TRAINING_H

#include <vector>
#include <string>

#include "cuda_interface.h"

#include "tree.h"
#include "utils.h"
#include "model.h"

typedef struct _trainingOptions {
  int batchSize;
  int numCycles;
  float learningRate;
  float regSoftmaxW;
  float regTransformW;
  float regTransformV;
  float regWordVectors;
} sTrainingOptions_t;

class SentimentTraining {
  public:
    SentimentTraining(const std::string& trainPath, const std::string& devPath, 
      int wordDim, int numClasses);
    void train(sTrainingOptions_t options);
    ~SentimentTraining();

  private:
    void trainBatch(int startIndex, int endIndex, const sTrainingOptions_t& options,
      cParamMem_t& derivatives, cParamMem_t& adagradWts);
    void computeDerivatives(int startIndex, int endIndex, const sTrainingOptions_t& options,
      cParamMem_t& derivatives, float& value);
    void forwardPropagate(Tree* tree);
    void backPropagate(Tree* tree);

    std::vector<Tree*> mTrainTrees;
    std::vector<Tree*> mDevTrees;
    SentimentModel mModel;
};

#endif // TRAINING_H
