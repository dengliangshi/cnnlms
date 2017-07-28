// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "hidden.h"

// --------------------------------------------------------Global Strings----------------------------------------------------


// ------------------------------------------------------------Main----------------------------------------------------------
HiddenLayers::HiddenLayers()
{
    objFNN = NULL;
    objRNN = NULL;
    objLSTM = NULL;
}

HiddenLayers::~HiddenLayers()
{
    if(objFNN != NULL) delete [] objFNN;
    if(objRNN != NULL) delete [] objRNN;
    if(objLSTM != NULL) delete [] objLSTM;
}

void HiddenLayers::InitModel(int inputSize, int layerNum, int* layerSize, int* layerName)
{
    intInputSize = inputSize;
    intLayerNum = layerNum;
    intLayerName = layerName;
    intLayerSize = layerSize;
    objFNN = new FeedForwardNN [intLayerNum];
    objRNN = new RecurrentNN [intLayerNum];
    objLSTM = new LSTMNN [intLayerNum];
    objBiRNN = new BiRecurrentNN [intLayerNum];
    InitLayer(0, intLayerName[0], intInputSize, intLayerSize[0]);
    if( intLayerNum > 1)
    {
        for(int i=1; i<intLayerNum; i++)
        {
            InitLayer(i, intLayerName[i],
                intLayerSize[i-1], intLayerSize[i]);
        }
    }
}

void HiddenLayers::InitLayer(int index, int modelName,
    int inputSize, int hiddenSize)
{
    switch(modelName)
    {
        case FNN:{
            objFNN[index].SetAcFun(intAcFun);
            objFNN[index].SetMaxLength(intMaxLen);
            objFNN[index].EnBias(bEnBias);
            objFNN[index].InitModel(inputSize, hiddenSize);
            break;
        }
        case RNN:{
            objRNN[index].SetAcFun(intAcFun);
            objRNN[index].SetMaxLength(intMaxLen);
            objRNN[index].EnBias(bEnBias);
            objRNN[index].InitModel(inputSize, hiddenSize);
            break;
        }
        case LSTM:{
            objLSTM[index].SetAcFun(intAcFun);
            objLSTM[index].SetGateFun(intGateFun);
            objLSTM[index].SetMaxLength(intMaxLen);
            objLSTM[index].EnBias(bEnBias);
            objLSTM[index].InitModel(inputSize, hiddenSize);
            break;
        }
        case BiRNN:{
            objBiRNN[index].SetAcFun(intAcFun);
            objBiRNN[index].SetMaxLength(intMaxLen);
            objBiRNN[index].EnBias(bEnBias);
            objBiRNN[index].InitModel(modelName, inputSize, hiddenSize);
            break;
        }
        case BiLSTM:{
            objBiRNN[index].SetAcFun(intAcFun);
            objBiRNN[index].SetGateFun(intGateFun);
            objBiRNN[index].SetMaxLength(intMaxLen);
            objBiRNN[index].EnBias(bEnBias);
            objBiRNN[index].InitModel(modelName, inputSize, hiddenSize);
            break;
        }
        default:{
            objLSTM[index].SetAcFun(intAcFun);
            objLSTM[index].SetGateFun(intGateFun);
            objLSTM[index].SetMaxLength(intMaxLen);
            objLSTM[index].EnBias(bEnBias);
            objLSTM[index].InitModel(inputSize, hiddenSize);
        }
    }
}

 void HiddenLayers::ResetMaxLength(int maxLength)
 {
    for(int i=0; i<intLayerNum; i++)
    {
        if(objFNN[i].isActive())
        {
            objFNN[i].ResetMaxLength(maxLength);
        }
        if(objRNN[i].isActive())
        {
            objRNN[i].ResetMaxLength(maxLength);
        }
        if(objLSTM[i].isActive())
        {
            objLSTM[i].ResetMaxLength(maxLength);
        }
        if(objBiRNN[i].isActive())
        {
            objBiRNN[i].ResetMaxLength(maxLength);
        }
    }
 }

weight* HiddenLayers::Run(weight *input, int length)
{
    weight *x = input;
    for(int i=0; i<intLayerNum; i++)
    {
        if(objFNN[i].isActive())
        {
            x = objFNN[i].Run(x, length);
        }
        if(objRNN[i].isActive())
        {
            x = objRNN[i].Run(x, length);
        }
        if(objLSTM[i].isActive())
        {
            x = objLSTM[i].Run(x, length);
        }
        if(objBiRNN[i].isActive())
        {
            x = objBiRNN[i].Run(x, length);
        }
    }
    return x;
}

void HiddenLayers::Update(double dAlpha, double dBeta)
{
    for(int i=intLayerNum-1; i>=0; i--)
    {
        if(objFNN[i].isActive())
        {
            objFNN[i].Update(dAlpha, dBeta);
        }
        if(objRNN[i].isActive())
        {
            objRNN[i].Update(dAlpha, dBeta);
        }
        if(objLSTM[i].isActive())
        {
            objLSTM[i].Update(dAlpha, dBeta);
        }
        if(objBiRNN[i].isActive())
        {
            objBiRNN[i].Update(dAlpha, dBeta);
        }
    }
}

void HiddenLayers::SaveModel(FILE *fout)
{
    for(int i=0; i<intLayerNum; i++)
    {
        if(objFNN[i].isActive())
        {
            objFNN[i].SaveModel(fout);
        }
        if(objRNN[i].isActive())
        {
            objRNN[i].SaveModel(fout);
        }
        if(objLSTM[i].isActive())
        {
            objLSTM[i].SaveModel(fout);
        }
        if(objBiRNN[i].isActive())
        {
            objBiRNN[i].SaveModel(fout);
        }
    }
}


void HiddenLayers::LoadModel(FILE *fin)
{
    for(int i=0; i<intLayerNum; i++)
    {
        if(objFNN[i].isActive())
        {
            objFNN[i].LoadModel(fin);
        }
        if(objRNN[i].isActive())
        {
            objRNN[i].LoadModel(fin);
        }
        if(objLSTM[i].isActive())
        {
            objLSTM[i].LoadModel(fin);
        }
        if(objBiRNN[i].isActive())
        {
            objBiRNN[i].LoadModel(fin);
        }
    }
}