#include "utils.h"

#include <fstream>

std::vector<Tree*>
SentimentUtils::readTrees(const std::string& path) {
    std::vector<Tree*> trees;
    std::ifstream fs(path.c_str());
    std::string line;
    while(std::getline(fs, line)) {
        //printf("parsing line: %s\n", line.c_str());
        trees.push_back(new Tree(line));
    }
    return std::move(trees);
}

void
SentimentUtils::cleanupTrees(std::vector<Tree*> trees) {
    for (auto tree: trees) {
        tree->cleanUp();
    }
}

void
SentimentUtils::getLeafWords(Tree* tree, std::set<std::string>& words) {
    if (tree->mIsLeaf) {
        //printf("%s\n", tree->mLabel.c_str());
        if (!tree->mLabel.empty()) {
            words.insert(tree->mLabel);
        }
    } else {
        if (tree->mLeftChild != nullptr) getLeafWords(tree->mLeftChild, words);
        if (tree->mRightChild != nullptr) getLeafWords(tree->mRightChild, words);
    }
}

void
SentimentUtils::assignLeafWordIds(Tree* tree,
  const std::map<std::string, int>& wordToIds, int unseenWordId) {
    if (tree->mIsLeaf) {
        if (!tree->mLabel.empty()) {
            auto it = wordToIds.find(tree->mLabel);
            if (it != wordToIds.end()) {
                tree->wordId = it->second;
            } else {
                tree->wordId = unseenWordId;
            }
        }
    } else {
        if (tree->mLeftChild != nullptr) assignLeafWordIds(tree->mLeftChild, wordToIds, unseenWordId);
        if (tree->mRightChild != nullptr) assignLeafWordIds(tree->mRightChild, wordToIds, unseenWordId);
    }
}