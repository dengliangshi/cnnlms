// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "nn.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
#ifndef _RECURRENTNN_H_
#define _RECURRENTNN_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class RecurrentNN: public NeuralNetwork{
public:
    RecurrentNN(): NeuralNetwork(){}
    ~RecurrentNN(){}

    // feedforward propagetion
    weight* Run(weight *input, int length);
    // update each parameters according to its error gradient
    void Update(double dAlpha, double dBeta);
};

#endif