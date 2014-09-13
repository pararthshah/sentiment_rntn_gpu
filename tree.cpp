#include "tree.h"

#include <cstdlib>
#include <algorithm>
#include <fstream>

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

    mGoldClass = std::atoi(treeStr.substr(1, 2).c_str());
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
Tree::getLeafWords(std::set<std::string>& words) {
    if (mIsLeaf) {
        //printf("%s\n", tree->mLabel.c_str());
        if (!mLabel.empty()) {
            words.insert(mLabel);
        }
    } else {
        if (mLeftChild != NULL) mLeftChild->getLeafWords(words);
        if (mRightChild != NULL) mRightChild->getLeafWords(words);
    }
}

void
Tree::assignNodeVectorsAndId(const std::map<std::string, int>& wordToIds, int unseenWordId,
    unsigned int wordDim, unsigned int numClasses) {
    if (mIsLeaf) {
        std::map<std::string, int>::const_iterator it = wordToIds.find(mLabel);
        if (it != wordToIds.end()) {
            mWordId = it->second;
        } else {
            mWordId = unseenWordId;
        }
    } else {
        // CudaInterface::allocMem(&mPredictedClassDist_d, numClasses);
        // CudaInterface::allocMem(&mNodeVector_d, wordDim);
        if (mLeftChild != NULL)
            mLeftChild->assignNodeVectorsAndId(wordToIds, unseenWordId, wordDim, numClasses);
        if (mRightChild != NULL)
            mRightChild->assignNodeVectorsAndId(wordToIds, unseenWordId, wordDim, numClasses);
    }   
}

void
Tree::cleanUp() {
    if (mLeftChild != NULL) {
        mLeftChild->cleanUp();
        mLeftChild = NULL;
    }
    if (mRightChild != NULL) {
        mRightChild->cleanUp();
        mRightChild = NULL;
    }
    // if (mNodeVector_d != NULL) {
    //     CudaInterface::freeMem(mNodeVector_d);
    //     mNodeVector_d = NULL;
    // }
    // if (mPredictedClassDist_d != NULL) {
    //     CudaInterface::freeMem(mPredictedClassDist_d);
    //     mPredictedClassDist_d = NULL;
    // }
}

void
Tree::readTrees(std::vector<Tree*>& trees, const std::string& path) {
    std::ifstream fs(path.c_str());
    std::string line;
    while(std::getline(fs, line)) {
        //printf("parsing line: %s\n", line.c_str());
        trees.push_back(new Tree(line));
    }
}

void
Tree::getAllLeafWords(std::vector<Tree*>& trees, std::set<std::string>& words) {
    for (int i = 0; i < trees.size(); i++) {
        trees[i]->getLeafWords(words);
    }
}

void
Tree::assignAllNodeVectorsAndId(std::vector<Tree*>& trees, const std::map<std::string, int>& wordToIds,
  int unseenWordId, unsigned int wordDim, unsigned int numClasses) {
    for (int i = 0; i < trees.size(); i++) {
        trees[i]->assignNodeVectorsAndId(wordToIds, unseenWordId, wordDim, numClasses);
    }
}

void
Tree::cleanupTrees(std::vector<Tree*>& trees) {
    for (int i = 0; i < trees.size(); i++) {
        trees[i]->cleanUp();
    }
}
