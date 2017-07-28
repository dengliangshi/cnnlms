// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library
#include <vector>
#include <ctime>

// Third-party Libraries


// User Define Module
#include "hidden.h"
#include "vocab.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
#ifndef _NNLM_H_
#define _NNLM_H_

// ------------------------------------------------------------Main----------------------------------------------------------
class NNLM{
private:
    string strVersion;       // version of this toolkit
    string strModelName;     // specify a name for this model
    // input files
    string strTrainFiles;    // directory of training files
    string strValidFiles;    // directory of validation files
    string strTestFiles;     // directory of test files
    // output files
    string strOutputPath;    // directory of output fiels
    string strLogFile;       // log file
    string strModelFile;     // file for saving model
    // marks for special files
    string strStartMark;     // mark for start of a sentence
    string strEndMark;       // mark for end of a sentence
    string strUnknown;       // mark for out of vocabulary words
    // paramters for language model
    int intVectorDim;        // dimension of word's feature vector
    int intGramOrder;        // grammer order for feedforward NNLM
    int intInputSize;        // the size of input vector for neural network
    int intClassOutput;      // size of output layer for word classes
    int intClassSize;        // the number of word classes
    int intClassAssign;      // word class assignment algorithm
    int intClassLayer;       // number of hierarchical class layer
    int intVocabSize;        // size of vocabulary
    int intTotalWords;       // total number of words in training data set
    int intHiddenSize;       // size of the last hidden layer
    int intLayerNum;         // number of hidden layers
    int intMaxLen;           // maximum of sequence's length
    int intFirstLayer;       // name of first hidden layer
    int *intLayerName;       // name of NNM for each hidden layer
    int *intLayerSize;       // the number of nodes in each hidden layer
    int intAcFun;            // activation function for hidden layers
    int intGateFun;          // activation function for gates in LSTM
    int intFileType;         // file type, text or binary
    int intInputUnit;        // the input level, 0 for word and 1 for character
    int intIterNum;          // maximum number of iteration
    int intRandomSeed;       // seed for random generator
    int intLength;           // length of input sequence
    int intWordNum;          // count the number of words from data set
    bool bEnBias;            // if enable bias terms in hidden layer
    bool bEnDirect;          // if enable direct connections
    bool bEnCache;           // if enable caching
    bool bEnDynamic;         // if ebable dynamic model
    bool bEnReverse;         // if reverse word order
    bool bEnTrainMode;       // if under training mode
    bool bEnTestMode;        // if under test mode
    bool bEnHierarchies;     // hierarchical word classes
    double dAlpha;           // learning rate
    double dBeta;            // regularization parameters
    double dMinRate;         // minimum improve rate on validation data
    double dLogP;            // logarithm probability
    // martrixes and vectors
    double *C;               // projection matrix, feacture vectors
    double *V;               // weight matrix in output layer for words
    double *Vc;              // weight matrix in output layer for classes
    double *M;               // weight matrix of direct connection for words
    double *Mc;              // weight matrix of direct connection for class
    double *d;               // bias vector in output layer for words
    double *dc;              // bias vector in output layer for classes
    // backup of weight matrixes
    double *Vb;              // backup of weight matrix V
    double *Vcb;             // backup of weight matrix Vc
    double *Mb;              // backup of weight matrix M
    double *Mcb;             // backup of weight matrix Mc
    // sequence
    weight *x;               // input vectors of hidden layers
    weight *s;               // output of hidden layers
    weight *y;               // sequence of NNLM's output for words
    weight *c;               // sequence of NNLM's output for word classes
    index unKnownIndex;      // index for unknown words
    index *wIndexs;          // index and class of words
    // objects 
    HiddenLayers objHidden;  // object of hidden layers
    Vocab objVocab;          // object of vocabulary
    

public:
    NNLM();
    ~NNLM();

    // generate a random number in given range
    double Random(double reLower, double reUpper);
    // set model name
    void SetModelName(string modelName){ strModelName = modelName;}
    // set model file
    void SetModelFile(string modelFile){ strModelFile = modelFile;}
    // set training file(s)
    void SetTrainFiles(string trainFiles){strTrainFiles = trainFiles;}
    // set validation file(s)
    void SetValidFiles(string validFiles){strValidFiles = validFiles;}
    // set training file(s)
    void SetTestFiles(string testFiles){strTestFiles = testFiles;}
    // set output path
    void SetOutputPath(string outputPath){strOutputPath = outputPath;}
    // set the marks for seoeical words
    void SetMarks(string startMark, string endMark, string unKnown);
    // set dimesion of feature vector
    void SetVectorDim(int vectorDim){ intVectorDim = vectorDim;}
    // set grammer order for feedforward NNLM
    void SetGramOrder(int gramOrder){ intGramOrder = gramOrder;}
    // set the number of word classes
    void SetClassSize(int classSize){ intClassSize = classSize;}
    // set the word class assignment algorithm
    void SetClassAssign(int classAssign){ intClassAssign = classAssign;}
    // set the number of hierarchical class layer
    void SetClassLayer(int classLayer){ intClassLayer = classLayer;}
    // set the size of vocabulary
    void SetVocabSize(int vocabSize){ intVocabSize=vocabSize;}
    // set activation function for hidden layers
    void SetAcFun(int acFun){ intAcFun = acFun;}
    // set activation function for gates in LSTM
    void SetGateFun(int gateFun){ intGateFun = gateFun;};
    // set file type
    void SetFileType(int fileType) { intFileType = fileType;}
    // set input unit
    void SetInputUnit(int inputUnit) { intInputUnit = inputUnit;}
    // set parameters for hidden layers
    void SetHiddenLayers(int layerNum, int *modelName, int *hiddenSize);
    // set maximum number of iteration
    void SetIterNum(int iterNum){ intIterNum = iterNum;}
    // set seed for random generator
    void SetRandomSeed(int randomSeed){ intRandomSeed = randomSeed;}
    // enable or disable bias terms
    void EnBias(bool enBias){ bEnBias = enBias;}
    // enable or disable diect connections
    void EnDirect(bool enDirect){ bEnDirect = enDirect;}
    // enable or disable caching
    void EnCache(bool enCache){ bEnCache = enCache;}
    // enable or disable dynamic model
    void EnDynamic(bool enDynamic){ bEnDynamic = enDynamic;}
    // reverse word order or not
    void EnReverse(bool enReverse){ bEnReverse = enReverse;}
    // enable training mode
    void EnTrainMode(bool enTrainMode){ bEnTrainMode = enTrainMode;}
    // enable test mode
    void EnTestMode(bool enTestMode){ bEnTestMode = enTestMode;}
    // set learning rate
    void SetAlpha(double Alpha){ dAlpha = Alpha;}
    // set regularization parameters
    void SetBeta(double Beta){ dBeta = Beta;}
    // set the minimun of improvement rate on validation data
    void SetMinRate(double minRate){ dMinRate = minRate;}
    // set maximum length of sequence
    void SetMaxLength(int maxLength){ intMaxLen = maxLength;}
    // get local time stamp
    void GetLocalTime(char* strTime);
    // initialize the nodes
    void InitModel();
    // initialize log file
    void InitLog();
    // activation function tanh
    void Softmax(weight *x, int lowerIndex, int upperIndex);
    // read sentence from a file
    void ReadSentence(FILE *fin);
    // compute error gradient of each parameters
    void Run();
    // reset maximum length of sequence
    void ResetMaxLength();
    // update each parameters according to its error gradient
    void Update();
    // train this neural network language model
    void TrainModel();
    // run model on validation data
    void ValidModel(FILE *finLog);
    // test the performance of NNLM
    void TestModel();
    // save the whole model
    void SaveModel(FILE *finLog);
    // load model from external file
    void LoadModel();
    // find delimiter in model file
    void GotoDelimiter(int intDelimiter, FILE *fin);
};

#endif