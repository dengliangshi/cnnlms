// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library
#include <cmath>

// Third-party Libraries


// User Define Module
#include "nodes.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class NeuralNetwork{
protected:
    int intInputSize;          // the dimension of input vector
    int intHiddenSize;         // the number of nodes in hidden layer
    int intLength;             // length of input sequence
    int intAcFun;              // activation function of hidden layer
    int intMaxLen;             // maximun of sequence's length
    bool bEnBias;              // if enable bias terms in hidden layer
    bool bActive;              // if this model is active
    weight *x;                 // input history
    weight *s;                 // history state of hidden layer
    Nodes objNodes;            // nodes in hidden layer

public:
    NeuralNetwork();
    ~NeuralNetwork();

    // set activation function for hidden layer
    void SetAcFun(int acFun){intAcFun = acFun;}
    // set the maximum length of sequence
    void SetMaxLength(int maxLength){intMaxLen = maxLength;}
    // enable or disenable bias terms
    void EnBias(bool enBias){bEnBias = enBias;}
    // reset the maximum length of sequence
    void ResetMaxLength(int maxLength);
    // initialize the nodes
    void InitModel(int inputSize, int hiddenSize);
    // tanh function
    double Tanh(double x);
    // derivation of function tanh
    double dTanh(double y);
    // hard tanh function
    double HardTanh(double x);
    // derivation of hard tanh function
    double dHardTanh(double x);
    // sigmoid function
    double Sigmoid(double x);
    // derivation of sigmoid function
    double dSigmoid(double y);
    // hard sigmoid function
    double HardSigmoid(double x);
    // derivation of hard sigmoud function
    double dHardSigmoid(double y);
    // ReLu, rectified linear units
    double ReLu(double x);
    //derivation of ReLu
    double dReLu(double y);
    // activation function
    double AcFun(double x, int acFun);
    // derivation of activation function
    double dAcFun(double y, int acFun);
    // if this model is active
    bool isActive(){ return bActive;}
    // save model
    void SaveModel(FILE *fout){objNodes.SaveModel(fout);}
    // load model
    void LoadModel(FILE *fin){objNodes.LoadModel(fin);}
};

#endif