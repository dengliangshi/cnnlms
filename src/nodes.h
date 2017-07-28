// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library
#include <cmath>

// Third-party Libraries


// User Define Module
#include "types.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
#ifndef _NODES_H_
#define _NODES_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class Nodes{
private:
    int intInputSize;             // the dimension of input vector
    int intHiddenSize;            // the number of nodes in hidden layer
    bool bEnBias;                 // if enable bias terms in hidden layer

public:
    weight *U;                    // weight matrix U
    weight *W;                    // weight matrix W
    weight *V;                    // weight matrix V
    weight *b;                    // bias terms b

public:
    Nodes();
    ~Nodes();

    // initialize the nodes
    void InitModel(int inputSize, int hiddenSize, bool enBias);
    // update each parameters according to its error gradient
    void Update(double dAlpha, double dBeta);
    // generate a random number in given range
    double Random(double dLower, double dUpper);
    // save model
    void SaveModel(FILE *fout);
    // load model
    void LoadModel(FILE *fin);
    // find delimiter
    void GotoDelimiter(int intDelimiter, FILE *fin);
};

#endif