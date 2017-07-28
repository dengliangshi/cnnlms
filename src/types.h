// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library
#include <map>
#include <vector>
#include <string>

// Third-party Libraries


// User Define Module


// --------------------------------------------------------Global Strings----------------------------------------------------
using namespace std;

#ifndef _TYPES_H_
#define _TYPES_H_


// ------------------------------------------------------------Main----------------------------------------------------------
// constants for the name neural netword model
#define FNN    0
#define RNN    1
#define LSTM   2
#define BiRNN  3
#define BiLSTM 4

//constants for activation functions
#define TANH         0
#define HARDTANH     1
#define SIGMOID      2
#define HARDSIGMOID  3
#define RELU         4
#define GAUSSIAN     5

// element of weight matrix
struct weight{
    double re;
    double er;
};

// words in vocabulary
struct word{
    string strName;
    int intIndex;
    int intClass;
    int intFreq;
    int *intVector;
};

// index and class for words
struct index{
    int intIndex;
    int intClass;
    int *intVector;
};

// type for vocabulary
typedef map<string, word> dict;

// type for sequence of vectors
typedef vector<weight *> sequence;

// type for word index
typedef vector<index> wordindex;

#endif