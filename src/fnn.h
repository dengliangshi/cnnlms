// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "nn.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
#ifndef _FEEDFORWARDNN_H_
#define _FEEDFORWARDNN_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class FeedForwardNN: public NeuralNetwork
{
public:
    FeedForwardNN(): NeuralNetwork(){}
    ~FeedForwardNN(){}

    // feedforward propagetion
    weight* Run(weight *input, int length);
    // update each parameters according to its error gradient
    void Update(double dAlpha, double dBeta);
};

#endif