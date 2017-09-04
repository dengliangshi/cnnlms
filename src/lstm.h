// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "nn.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
#ifndef _LSTM_H_
#define _LSTM_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class LSTMNN: public NeuralNetwork{
protected:
    double* ig;                // state of input gate
    double* fg;                // state of forget gate
    double* og;                // state of output gate
    double* g;                 // internal hidden states
    double* h;                 // hidden states
    weight* c;                 // history of internal memory
    Nodes iGate;               // nodes of input gate
    Nodes fGate;               // nodes of forget gate
    Nodes oGate;               // nodes of output gate
    int intGateFun;            // activation function of gates
    
public:
    LSTMNN(): NeuralNetwork()
    {
        c = NULL;
        h = NULL;
        g = NULL;
        ig = NULL;
        fg = NULL;
        og = NULL;
    }
    ~LSTMNN()
    {
        if(c != NULL) delete [] c;
        if(h != NULL) delete [] h;
        if(g != NULL) delete [] g;
        if(ig != NULL) delete [] ig;
        if(fg != NULL) delete [] fg;
        if(og != NULL) delete [] og;
    }

    // set activation function for gates in LSTM
    void SetGateFun(int gateFun){ intGateFun = gateFun;}
    // reset the maximum length of sequence
    void ResetMaxLength(int maxLength);
    // initialize the nodes
    void InitModel(int inputSize, int hiddenSize);
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