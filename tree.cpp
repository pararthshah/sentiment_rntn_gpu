#include "tree.h"

#include <cstdlib>
#include <algorithm>

#include "cuda_interface.h"

Tree::Tree(const std::string& treeStr) {
    Tree();
    //printf("treeStr: %s\n", treeStr.c_str());
    int strLength = treeStr.size();
    if (strLength < 4) {
        printf("malformed '%s'\n", treeStr.c_str());
        std::exit(1);
    }

    if (!(treeStr[0] == '(' && treeStr[strLength-1] == ')')) {
        printf("error reading '%s'\n", treeStr.c_str());
        std::exit(1);
    }

    mGoldClass = std::stoi(treeStr.substr(1, 2));
    if (treeStr[3] != '(') {
        mIsLeaf = true;
        mLabel = treeStr.substr(3, strLength-4);
        std::transform(mLabel.begin(), mLabel.end(), mLabel.begin(), ::tolower);
        //printf("label=%s\n",mLabel.c_str());
    } else {
        mIsLeaf = false;
        // find split
        int splitIndex = -1;
        int currDepth = 1;
        for (int i = 4; i < strLength; i++) {
            if (treeStr[i] == '(') currDepth++;
            if (treeStr[i] == ')') currDepth--;
            if (currDepth == 0) {
                splitIndex = i+1;
                break;
            }
        }
        if (!(splitIndex > 3 && splitIndex < strLength-2)) {
            printf("error parsing '%s'\n", treeStr.c_str());
            std::exit(1);
        }
        mLeftChild = new Tree(treeStr.substr(3, splitIndex-3));
        mRightChild = new Tree(treeStr.substr(splitIndex+1, strLength-splitIndex-2));
    }
}

void
Tree::cleanUp() {
    if (mLeftChild != nullptr) {
        mLeftChild->cleanUp();
        mLeftChild = nullptr;
    }
    if (mRightChild != nullptr) {
        mRightChild->cleanUp();
        mRightChild = nullptr;
    }
    if (mNodeVector_d != nullptr) {
        CudaInterface::freeMem(mNodeVector_d);
        mNodeVector_d = nullptr;
    }
    if (mPredictedClassDist_d != nullptr) {
        CudaInterface::freeMem(mPredictedClassDist_d);
        mPredictedClassDist_d = nullptr;
    }
}