// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "rnn.h"
#include "lstm.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
#ifndef _BIRECURRENTNN_H_
#define _BIRNN_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class BiRecurrentNN{
protected:
    int intModelName;          // the name of nn model, birnn or bilstm
    int intInputSize;          // the dimension of input vector
    int intHiddenSize;         // the number of nodes in hidden layer
    int intHalfSize;           // half of hidden size
    int intLength;             // length of input sequence
    int intAcFun;              // activation function of hidden layer
    int intGateFun;            // activation function for gates in LSTM
    int intMaxLen;             // maximun of sequence's length
    bool bEnBias;              // if enable bias terms in hidden layer
    bool bActive;              // if this model is active
    weight *x;                 // input history
    weight *s;                 // history state of hidden layer
    weight *fx;                // input history for feed-forward layer
    weight *bx;                // input history for back-forward layer
    weight *fs;                // history state of feed-forward layer
    weight *bs;                // history state of back-forward layer
    RecurrentNN FRNN;          // feed-forward recurrent nn layers
    RecurrentNN BRNN;          // back-forward recurrent nn layers
    LSTMNN FLSTM;              // feed-forward lstm recurrent nn layers
    LSTMNN BLSTM;              // back-forward lstm recurrent nn layers

public:
     BiRecurrentNN();
    ~BiRecurrentNN();

    // set activation function for hidden layer
    void SetAcFun(int acFun){ intAcFun = acFun;}
    // set activation function for gates
    void SetGateFun(int gateFun){ intGateFun = gateFun;}
    // set the maximum length of sequence
    void SetMaxLength(int maxLength){ intMaxLen = maxLength; }
    // enable or disenable bias terms
    void EnBias(bool enBias){ bEnBias = enBias;}
    // reset the maximum length of sequence
    void ResetMaxLength(int maxLength);
    // initialize the nodes
    void InitModel(int modelName, int inputSize, int hiddenSize);
    // if this model is active
    bool isActive(){ return bActive;}
    // feedforward propagetion
    weight* Run(weight *input, int length);
    // update each parameters according to its error gradient
    void Update(double dAlpha, double dBeta);
    // save model
    void SaveModel(FILE *fout);
    // load model
    void LoadModel(FILE *fin);
};

#endif