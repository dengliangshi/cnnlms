// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "nn.h"


// --------------------------------------------------------Global Strings----------------------------------------------------


// ------------------------------------------------------------Main----------------------------------------------------------
NeuralNetwork::NeuralNetwork()
{
    x = NULL;
    s = NULL;
    bActive = 0;
}

NeuralNetwork::~NeuralNetwork()
{
    if(s != NULL) delete [] s;
}

void NeuralNetwork::InitModel(int inputSize, int hiddenSize)
{
    bActive = 1;
    intLength = 0;
    intInputSize = inputSize;
    intHiddenSize = hiddenSize;

    s = new weight [intMaxLen * intHiddenSize];
    for(int i=0; i<intHiddenSize; i++)
    {
        s[i].re = 0.1; s[i].er = 0;
    }
    objNodes.InitModel(intInputSize, intHiddenSize, bEnBias);
}

void NeuralNetwork::ResetMaxLength(int maxLength)
{
    if(intMaxLen < maxLength)
    {
        intMaxLen = maxLength;
        if(s != NULL) delete [] s;
        s = new weight [intMaxLen * intHiddenSize];
        for(int i=0; i<intHiddenSize; i++)
        { 
            s[i].re = 0.1; s[i].er = 0;
        }
    }
}

double NeuralNetwork::Tanh(double x)
{
    return tanh(x);
}

double NeuralNetwork::dTanh(double y)
{
    return 1.0 - y * y;
}

double NeuralNetwork::HardTanh(double x)
{
    if(x < -1.0)
    {
        return -1.0;
    }
    if(x > 1.0)
    {
        return 1.0;
    }
    return x;
}

double NeuralNetwork::dHardTanh(double y)
{
    if(y >= -1.0 && y <= 1.0)
    {
        return 1.0;
    }
    return 0;
}

double NeuralNetwork::Sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::dSigmoid(double y)
{
    return (1.0 - y) * y;
}

double NeuralNetwork::HardSigmoid(double x)
{
    double y;

    y = (x + 1.0) / 2.0;
    if(y < 0)
    {
        return 0;
    }
    if(y > 1.0)
    {
        return 1.0;
    }
    return y;
}

double NeuralNetwork::dHardSigmoid(double y)
{
    if(y > 0 && y < 1.0)
    {
        return 0.5;
    }
    return 0;
}

double NeuralNetwork::ReLu(double x)
{
    if(x < 0)
    {
        return 0;
    }
    return x;
}

double NeuralNetwork::dReLu(double y)
{
    if(y > 0)
    {
        return 1.0;
    }
    else
    {
        return 0;
    }
}

double NeuralNetwork::AcFun(double x, int acFun)
{
    switch(acFun)
    {
        case 0: return Tanh(x);
        case 1: return HardTanh(x);
        case 2: return Sigmoid(x);
        case 3: return HardSigmoid(x);
        case 4: return ReLu(x);
        default: return Tanh(x);
    }
}

double NeuralNetwork::dAcFun(double y, int acFun)
{
    switch(acFun)
    {
        case 0: return dTanh(y);
        case 1: return dHardTanh(y);
        case 2: return dSigmoid(y);
        case 3: return dHardSigmoid(y);
        case 4: return dReLu(y);
        default: return dTanh(y);
    }
}