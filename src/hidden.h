// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library
#include <cmath>

// Third-party Libraries


// User Define Module
#include "fnn.h"
#include "rnn.h"
#include "lstm.h"
#include "birnn.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
#ifndef _HIDDEN_H_
#define _HIDDEN_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class HiddenLayers{
private:
    int *intLayerName;         // name of hidden layers
    int *intLayerSize;         // size of hidden layers
    int intLayerNum;           // number of hidden layers
    int intInputSize;          // the dimension of input vector
    int intAcFun;              // activation function for hidden layers
    int intGateFun;            // activation function for gates in LSTM
    int intMaxLen;             // maximum of sequence's length
    bool bEnBias;              // if enable bias terms in hidden layer
    FeedForwardNN *objFNN;     // feedforward nn layers
    RecurrentNN *objRNN;       // recurrent nn layers
    LSTMNN *objLSTM;           // lstm recurrent nn layers
    BiRecurrentNN *objBiRNN;   // bidirectional rnn layers

public:
    HiddenLayers();
    ~HiddenLayers();
    
    // set activation function for hidden layer
    void SetAcFun(int acFun){ intAcFun = acFun;}
    // set activation function for gates in LSTM
    void SetGateFun(int gateFun){ intGateFun = gateFun;};
    // enable or disable bias terms
    void EnBias(bool enBias){ bEnBias = enBias;}
    // set maximum length of sequence
    void SetMaxLength(int maxLength){ intMaxLen = maxLength;}
    // initialize the model
    void InitModel(int inputSize, int layerNum, int* hiddenSize, int* modelName);
    // initialize each layer
    void InitLayer(int index, int modelName, int inputSize, int hiddenSize);
    // reset maximum length of sequence
    void ResetMaxLength(int maxLength);
    // feedforward propagation
    weight* Run(weight *x, int length);
    // update each parameters according to its error gradient
    void Update(double dAlpha, double dBeta);
    // save model
    void SaveModel(FILE *fout);
    // load model
    void LoadModel(FILE *fin);
};

#endif