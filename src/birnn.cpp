// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "birnn.h"


// --------------------------------------------------------Global Strings----------------------------------------------------


// ------------------------------------------------------------Main----------------------------------------------------------
 BiRecurrentNN::BiRecurrentNN()
{
    s = NULL;
    fx = NULL;
    bx = NULL;
    bActive = 0;
}

 BiRecurrentNN::~BiRecurrentNN()
{
    if(s != NULL) delete [] s;
    if(fx != NULL) delete [] fx;
    if(bx != NULL) delete [] bx;
}

void BiRecurrentNN::ResetMaxLength(int maxLength)
{
    if(intMaxLen < maxLength)
    {
        intMaxLen = maxLength;
        if(s != NULL) delete [] s;
        if(fx != NULL) delete [] fx;
        if(bx != NULL) delete [] bx;
        s = new weight [intMaxLen*intHiddenSize];
        fx = new weight [intMaxLen*intInputSize];
        bx = new weight [intMaxLen*intInputSize];
        for(int i=0; i<intHiddenSize; i++)
        { 
            s[i].re = 0.1; s[i].er = 0;
        }
        if(intModelName == BiRNN)
        {
            FRNN.ResetMaxLength(intMaxLen);
            BRNN.ResetMaxLength(intMaxLen);
        }
        if(intModelName == BiLSTM)
        {
            FLSTM.ResetMaxLength(intMaxLen);
            BLSTM.ResetMaxLength(intMaxLen);
        }
    }
}

void BiRecurrentNN::InitModel(int modelName, int inputSize, int hiddenSize)
{
    bActive = 1;
    intModelName = modelName;
    intInputSize = inputSize;
    intHiddenSize = hiddenSize;
    intHalfSize = intHiddenSize/2;

    s = new weight [intMaxLen*intHiddenSize];
    fx = new weight [intMaxLen*intInputSize];
    bx = new weight [intMaxLen*intInputSize];
    for(int i=0; i<intHiddenSize; i++)
    {
        s[i].re = 0.1; s[i].er = 0;
    }

    if(intModelName == BiRNN)
    {
        FRNN.SetAcFun(intAcFun);
        FRNN.SetMaxLength(intMaxLen);
        FRNN.EnBias(bEnBias);
        FRNN.InitModel(intInputSize, intHalfSize);
        BRNN.SetAcFun(intAcFun);
        BRNN.SetMaxLength(intMaxLen);
        BRNN.EnBias(bEnBias);
        BRNN.InitModel(intInputSize, intHalfSize);
    }
    if(intModelName == BiLSTM)
    {
        FLSTM.SetAcFun(intAcFun);
        FLSTM.SetGateFun(intGateFun);
        FLSTM.SetMaxLength(intMaxLen);
        FLSTM.EnBias(bEnBias);
        FLSTM.InitModel(intInputSize, intHalfSize);
        BLSTM.SetAcFun(intAcFun);
        BLSTM.SetGateFun(intGateFun);
        BLSTM.SetMaxLength(intMaxLen);
        BLSTM.EnBias(bEnBias);
        BLSTM.InitModel(intInputSize, intHalfSize);
    }
}

weight* BiRecurrentNN::Run(weight *input, int length)
{
    x = input;
    intLength = length;

    for(int t=0; t<intLength; t++)
    {
        for(int i=0; i<intInputSize; i++)
        {
            fx[intInputSize*(t+1)+i].re = x[intInputSize*(t+1)+i].re;
            fx[intInputSize*(t+1)+i].er = 0; 
            bx[intInputSize*(intLength-t)+i].re = x[intInputSize*(t+1)+i].re;
            bx[intInputSize*(intLength-t)+i].er = 0;
        }
    }
    if(intModelName == BiRNN)
    {
        fs = FRNN.Run(fx, intLength);
        bs = BRNN.Run(bx, intLength);
    }
    if(intModelName == BiLSTM)
    {
        fs = FLSTM.Run(fx, intLength);
        bs = BLSTM.Run(bx, intLength);
    }
    for(int t=0; t<intLength; t++)
    {
        for(int i=0; i<intHalfSize; i++)
        {
            s[intHiddenSize*(t+1)+i].re = fs[intHalfSize*(t+1)+i].re;
            s[intHiddenSize*(t+1)+i].er = 0;
            s[intHiddenSize*(t+1)+intHalfSize+i].re = bs[intHalfSize*(intLength-t)+i].re;
            s[intHiddenSize*(t+1)+intHalfSize+i].er = 0;
        }
    }
    return s;
}

void BiRecurrentNN::Update(double dAlpha, double dBeta)
{
    for(int t=0; t<intLength; t++)
    {
        for(int i=0; i<intHalfSize; i++)
        {
            fs[intHalfSize*(t+1)].er = s[intHiddenSize*(t+1)+i].er;
            bs[intHalfSize*(intLength-t)+i].er = s[intHiddenSize*(t+1)+intHalfSize+i].er;
        }
    }
    if(intModelName == BiRNN)
    {
        FRNN.Update(dAlpha, dBeta);
        BRNN.Update(dAlpha, dBeta);
    }
    if(intModelName == BiLSTM)
    {
        FLSTM.Update(dAlpha, dBeta);
        BLSTM.Update(dAlpha, dBeta);
    }
    for(int t=0; t<intLength; t++)
    {
        for(int i=0; i<intInputSize; i++)
        {
            x[intInputSize*(t+1)+i].er += fx[intInputSize*(t+1)+i].er; 
            x[intInputSize*(t+1)+i].er += bx[intInputSize*(intLength-t)+i].er;
        }
    }
}

void BiRecurrentNN::SaveModel(FILE *fout)
{
    if(intModelName == BiRNN)
    {
        FRNN.SaveModel(fout);
        BRNN.SaveModel(fout);
    }
    if(intModelName == BiLSTM)
    {
        FLSTM.SaveModel(fout);
        BLSTM.SaveModel(fout);
    }
}

void BiRecurrentNN::LoadModel(FILE *fin)
{
    if(intModelName == BiRNN)
    {
        FRNN.LoadModel(fin);
        BRNN.LoadModel(fin);
    }
    if(intModelName == BiLSTM)
    {
        FLSTM.LoadModel(fin);
        BLSTM.LoadModel(fin);
    }
}