// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library
#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>

// Third-party Libraries


// User Define Module
#include "types.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
using namespace std;

#ifndef _VOCAB_H_
#define _VOCAB_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class Vocab{
private:
    int intVocabSize;             // the size of vocab
    int intWordCount;             // the number of words added into vocabulary
    int intTotalWords;            // total number of words in training data set
    int intClassSize;             // the size of word class
    int intClassAssign;           // word class assignment algorithm
    int intClassLayer;            // number of hierarchical word class layer
    int intTotalSize;             // total number of word classes
    int intInputUnit;             // the input level, 0 for word and 1 for character
    int intFileType;              // the format of training files, 0 for text and 1 for binary
    int *intSubClassSize;         // the number of words in each class
    bool bEnHierarchies;          // hierarchical word classes
    string strTrainFiles;         // the directory of traing files
    dict dictVocab;               // vocabulary list
    string strStartMark;          // end mark of a sentence
    string strEndMark;            // start mark of a sentence
    string strUnknown;            // string for unkown words

public:
    Vocab();
    ~Vocab();
    
    // set the number of word classes
    void SetClassSize(int classSize){ intClassSize = classSize;}
    // set the word class assignment algorithm
    void SetClassAssign(int classAssign){ intClassAssign = classAssign;}
    // set the number of hierarchical word class layer
    void SetClassLayer(int classLayer){ intClassLayer = classLayer;}
    // set the size of vocabulary
    void SetVocabSize(int vocabSize){ intVocabSize = vocabSize;}
    // set input level
    void SetInputUnit(int inputUnit) { intInputUnit = inputUnit;}
    // set the marks for seoeical words
    void SetMarks(string startMark, string endMark, string unKnown);
    // initialize the model
    void InitModel(string trainFiles, int fileType=0);
    // generate vocabulary from training files
    int Generate();
    // read a word from file
    string ReadWord(FILE *fin);
    // read a character from file
    string ReadChar(FILE *fin);
    // add word intio vocabulary
    void AddWord(string strWord);
    // sort the vocabuary accoding to each word's frequency
    void SortVocab();
    // assign each word in vocabulary with index
    void AssignIndex(word *wordList);
    // get the index of the given word
    void GetIndex(string strWord, index *iWord);
    // get the lower and upper index of the words in the given class
    void GetRange(int classIndex, int *lowerIndex, int *upperIndex);
    // the total number of words in traing data
    int GetTotalWords() { return intTotalWords;}
    // save vocabulary
    void SaveModel(FILE *fout);
    // load vocabulary
    void LoadModel(FILE *fin);
    // find delimiter
    void GotoDelimiter(int intDelimiter, FILE *fin);
};

#endif