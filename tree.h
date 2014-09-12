#ifndef TREE_H
#define TREE_H

#include <string>

class Tree {
  public:
    std::string mLabel;
    bool mIsLeaf;
    int mGoldClass;
    int mPredictedClass;
    float* mPredictedClassDist_d;

    int wordId;
    float* mNodeVector_d; // this is stored in device memory

    Tree* mLeftChild;
    Tree* mRightChild;

    Tree() {
        mLeftChild = nullptr;
        mRightChild = nullptr;
        mPredictedClassDist_d = nullptr;
        mNodeVector_d = nullptr;
        wordId = -1;
    }

    Tree(const std::string& treeStr);

    void cleanUp();
};

#endif // TREE_H
