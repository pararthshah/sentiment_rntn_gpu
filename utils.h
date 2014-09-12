#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <set>
#include <map>
#include <cstdlib>

#include "tree.h"

class SentimentUtils {
  public:
    static std::vector<Tree*> readTrees(const std::string& path);
    static void cleanupTrees(std::vector<Tree*> trees);
    static void getLeafWords(Tree* tree, std::set<std::string>& words);
    static void assignLeafWordIds(Tree* tree, const std::map<std::string, int>& wordToIds, int unseenWordId);
};

#endif // UTILS_H