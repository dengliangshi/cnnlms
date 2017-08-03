// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "fnn.h"


// --------------------------------------------------------Global Strings----------------------------------------------------
// s(t) = f(U*x(t) + b)

// ------------------------------------------------------------Main----------------------------------------------------------
weight* FeedForwardNN::Run(weight *input, int length)
{
    int intIndex;

    x = input;
    intLength = length;
    for(int t=0; t<intLength; t++)
    {
        for(int i=0; i<intHiddenSize; i++)
        {
            intIndex = intHiddenSize*(t+1) + i;
            s[intIndex].re = 0;
            s[intIndex].er = 0;
            for(int j=0; j<intInputSize; j++)
            {
                s[intIndex].re += objNodes.U[intInputSize*i+j].re * x[intInputSize*(t+1)+j].re;
            }
            if(bEnBias) {s[intIndex].re += objNodes.b[i].re;}
            if(s[intIndex].re > 50){ s[intIndex].re = 50; }
            if(s[intIndex].re < -50){ s[intIndex].re = -50; }
            s[intIndex].re = AcFun(s[intIndex].re, intAcFun);
        }
    }
    return s;
}

void FeedForwardNN::Update(double dAlpha, double dBeta)
{
    double dLdp;
    for(int t=0; t<intLength; t++)
    {
        for(int i=0; i<intHiddenSize; i++)
        {
            dLdp = s[intHiddenSize*(t+1)+i].er * dAcFun(s[intHiddenSize*(t+1)+i].re, intAcFun);
            for(int j=0; j<intInputSize; j++)
            {
                x[intInputSize*(t+1)+j].er += dLdp * objNodes.U[intInputSize*i+j].re;
                objNodes.U[intInputSize*i+j].er += dLdp * x[intInputSize*(t+1)+j].re;
            }
            if(bEnBias) {objNodes.b[i].er += dLdp;}
        }
    }
    objNodes.Update(dAlpha, dBeta);
}