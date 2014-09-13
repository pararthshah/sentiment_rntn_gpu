#ifndef TREE_H
#define TREE_H

#include <string>
#include <vector>
#include <set>
#include <map>

class Tree {
  public:
    std::string mLabel;
    bool mIsLeaf;
    int mGoldClass;
    int mPredictedClass;
    float* mPredictedClassDist_d;

    int mWordId;
    float* mNodeVector_d; // this is stored in device memory

    Tree* mLeftChild;
    Tree* mRightChild;

    Tree() {
        mLeftChild = NULL;
        mRightChild = NULL;
        mPredictedClassDist_d = NULL;
        mNodeVector_d = NULL;
        mWordId = -1;
    }

    Tree(const std::string& treeStr);

    void getLeafWords(std::set<std::string>& words);
    void assignNodeVectorsAndId(const std::map<std::string, int>& wordToIds, int unseenWordId,
      unsigned int wordDim, unsigned int numClasses);
    void cleanUp();

    static void readTrees(std::vector<Tree*>& trees, const std::string& path);
    static void getAllLeafWords(std::vector<Tree*>& trees, std::set<std::string>& words);
    static void assignAllNodeVectorsAndId(std::vector<Tree*>& trees, const std::map<std::string, int>& wordToIds,
      int unseenWordId, unsigned int wordDim, unsigned int numClasses);
    static void cleanupTrees(std::vector<Tree*>& trees);
};

#endif // TREE_H
