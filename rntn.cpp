#include <string>
#include <cstdlib>

#include "training.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("provide training file dir\n");
        std::exit(1);
    }
    std::string treeDir(argv[1]);
    std::string trainPath = treeDir + "/train.txt";
    std::string devPath = treeDir + "/dev.txt";
    
    // model parameters
    int wordDim = 32;
    int numClasses = 5;
    SentimentTraining trainer(trainPath, devPath, wordDim, numClasses);

    // training parameters
    sTrainingOptions_t options = {
        25,
        1,
        0.01,
        0.0001,
        0.001,
        0.001,
        0.0001
    };
    trainer.train(options);

    return 0;
}