// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library
#include <fstream>
#include <iostream>

// Third-party Libraries


// User Define Module
#include "nnlmlib.h"

// --------------------------------------------------------Global Strings----------------------------------------------------
using namespace std;

// ------------------------------------------------------------Main----------------------------------------------------------
int SearchArgs(string str, int argc, char **argv)
{
    for(int i=1; i<argc; i++)
    {
        if(str == argv[i])
        { 
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv)
{
    int intIndex;
    string strModelName = "";              // specify a name for this model
    string strModelFile = "";              // model file
    string strTrainFiles = "";             // directory of training files
    string strValidFiles = "";             // directory of validation files
    string strTestFiles = "";              // directory of test files
    string strOutputPath = "";             // directory of output fiels
    string strStartMark = "<s>";
    string strEndMark = "</s>";
    string strUnknown = "OOV";
    int intVectorDim = 100;
    int intGramOrder = 5;
    int intClassSize = 100;
    int intClassLayer = 1;
    int intClassAssign = 2;
    int intVocabSize = 1000000;
    int intHiddenSize = 100;
    int intLayerNum = 1;
    int intMaxLen = 100;
    int intFirstLayer = 0;
    int *intLayerName = NULL;
    int *intLayerSize = NULL;
    int intAcFun = 0;
    int intGateFun = 1;
    int intFileType = 0;
    int intInputUnit = 0;
    int intIterNum = 20;
    int intRandomSeed = 1;
    bool bEnCache = 0;
    bool bEnBias = 0;
    bool bEnDirect = 0;
    bool bEnDynamic = 0;
    bool bEnReverse = 0;
    bool bDebugMode = 0;
    double dAlpha = 0.01;
    double dBeta = 1e-3;
    double dMinRate = 1.003;
    bool bEnTrainMode = 0;
    bool bEnTestMode = 0;


    intLayerName = new int [intLayerNum];
    intLayerSize = new int [intLayerNum];
    intLayerName[0] = intFirstLayer;
    intLayerSize[0] = intHiddenSize;

    // options and arguments of model
    if (argc==2 and ((string)"-help" == argv[1]))
    {
        printf("Neural network language modeling toolkit v1.0\n");
        printf("Options and arguments:\n");
        printf("-alpha: learning rate, default is 0.01;\n");
        printf("-acfun: code for activation function in hidden layers, 0-tanh, 1-hard tanh,\n");
        printf("\t2-sigmoid, 3-hard sigmoid, 4- relu, default is 0;\n");
        printf("-beta: regularization parameters, default is 1.0e-6;\n");
        printf("-bias: enable or disable bias terms, 0-disable, 1-enable, default is 0;\n");
        printf("-cache: enable caching, information from previous sentence will be taken into account;\n");
        printf("-class-assign: the code for class assignment algorithm;\n");
        printf("-class-layer: the number of hierarchical class layers, default is 1;\n");
        printf("-class-size: number of word classes, default is 100;\n");
        printf("-debug: enable or disable debug model;\n");
        printf("-direct: enable or disable direct connections;\n");
        printf("-dynamic: update model's parameters during test if enable this option;\n");
        printf("-end-mark: mark for end of sentence, default is </s>;\n");
        printf("-gatefun: code for activation function of gates in LSTM, 0-tanh, 1-hard tanh,\n");
        printf("\t2-sigmoid, 3-hard sigmoid, 4- relu, default is 0;\n");
        printf("-help: get help information;\n");
        printf("-input-unit: code for inpurt level of model, 0-word, 1-character, default is 0;\n");
        printf("-iter: maximum iteration, default is 20;\n");
        printf("-layer-num: number of hidden layers, default is 1;\n");
        printf("-layer-name: code for each hidden layer's name, 0-FNN, 1-RNN, 2-LSTM, 3-BiRNN,\n");
        printf("\t4-BiLSTM, default is 0;\n");
        printf("-layer-size: the size of each hidden layer, default is 100;\n");
        printf("-max-len: maximum length of sentence, default is 100;\n");
        printf("-mini-improv: minimun improvement on validation data, default is 1.003;\n");
        printf("-model: given a model file when load model from external file;\n");
        printf("-name: specify a name for this model;\n");
        printf("-order: order of n-gram for feedforward nnlm, default is 5;\n");
        printf("-output: directory for output file(s);\n");
        printf("-reverse: reverse the order of words in input sentence;\n");
        printf("-seed: seed for random generator, default is 1;\n");
        printf("-start-mark: mark for start of sentence, default is <s>;\n");
        printf("-test: parent directory of test file(s), multiple files supported;\n");
        printf("-train: parent directory of training file(s), multiple files supported;\n");
        printf("-unknown: mark for words out of vocabulary, default is OOV;\n");
        printf("-valid: parent directory of validation file(s), multiple files supported;\n");
        printf("-vector-dim: dimension of word feature vector, default is 100;\n");   
        printf("-vocab-size: size of vocabulary which will be overwriten if the number of words\n");
        printf("\tfrom training data is less than this size , default is 100000;\n");
        return 0;
    }

    // enable debug model
    intIndex = SearchArgs("-debug", argc, argv);
    if (intIndex > 0)
    {
        bDebugMode = 1;
        if (bDebugMode > 0)
        {
            printf("Debug Mode: %d\n", bDebugMode);
        }
    }
    // specify a name for this model
    intIndex = SearchArgs("-name", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: model name not specified!\n");
            return 0;
        }
        strModelName = argv[intIndex+1];
        if (bDebugMode > 0)
        {
            printf("Model Name: %s\n", strModelName.c_str());
        }
    }
    // specify a model file
    intIndex = SearchArgs("-model", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: model file not specified!\n");
            return 0;
        }
        strModelFile = argv[intIndex+1];
        if (bDebugMode > 0)
        {
            printf("Model File: %s\n", strModelFile.c_str());
        }
    }
    // parameters for training files
    intIndex = SearchArgs("-train", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: training data files not specified!\n");
            return 0;
        }
        strTrainFiles = argv[intIndex+1];
        bEnTrainMode = 1;
        if (bDebugMode > 0)
        {
            printf("Training Files: %s\n", strTrainFiles.c_str());
        }
    }
    // parameters for validation files
    intIndex = SearchArgs("-valid", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: validation data files not specified!\n");
            return 0;
        }
        strValidFiles = argv[intIndex+1];
        if (bDebugMode > 0)
        {
            printf("Validation Files: %s\n", strValidFiles.c_str());
        }
    }
    // parameters for test files
    intIndex = SearchArgs("-test", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Test data files not specified!\n");
            return 0;
        }
        strTestFiles = argv[intIndex+1];
        bEnTestMode = 1;
        if (bDebugMode > 0)
        {
            printf("Test Files: %s\n", strTestFiles.c_str());
        }
    }
    // parameters for output files
    intIndex = SearchArgs("-output", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Path for output files not specified!\n");
            return 0;
        }
        strOutputPath = argv[intIndex+1];
        if (bDebugMode > 0)
        {
            printf("Path for Output Files: %s\n", strOutputPath.c_str());
        }
    }
    // parameters for start mark of sentence
    intIndex = SearchArgs("-start-mark", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Start mark of sentence not specified!\n");
            return 0;
        }
        strStartMark = argv[intIndex+1];
        if (bDebugMode > 0)
        {
            printf("Start Mark of Sentence: %s\n", strStartMark.c_str());
        }
    }
    // parameters for end mark of sentence
    intIndex = SearchArgs("-end-mark", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: End mark of sentence not specified!\n");
            return 0;
        }
        strEndMark = argv[intIndex+1];
        if (bDebugMode > 0)
        {
            printf("End Mark of Sentence: %s\n", strEndMark.c_str());
        }
    }
    // parameters for words out of vocabulary
    intIndex = SearchArgs("-unknown", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Mark of unknown words not specified!\n");
            return 0;
        }
        strUnknown = argv[intIndex+1];
        if (bDebugMode > 0)
        {
            printf("Mark of Unknown Words: %s\n", strUnknown.c_str());
        }
    }
    // parameters for the dimension of feature vector
    intIndex = SearchArgs("-vector-dim", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Dimension of word feactur vectors not specified!\n");
            return 0;
        }
        intVectorDim = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Dimension of Word Feactur Vectors: %d\n", intVectorDim);
        }
    }
    // parameters for order of n-gram
    intIndex = SearchArgs("-order", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: N-gram order not specified!\n");
            return 0;
        }
        intGramOrder = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("N-gram Order: %d\n", intGramOrder);
        }
    }
    // parameters for number of word classes
    intIndex = SearchArgs("-class-size", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Number of word classes not specified!\n");
            return 0;
        }
        intClassSize = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Number of Word Classes: %d\n", intClassSize);
        }
    }
    // parameters for class assignment algorithm
    intIndex = SearchArgs("-class-assign", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Code for class assignment algorithm not specified!\n");
            return 0;
        }
        intClassAssign = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Code for Class Assignment Algorithm: %d\n", intClassAssign);
        }
    }
    // parameters for hierarchical class layer
    intIndex = SearchArgs("-class-layer", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Number of hierarchical class layer not specified!\n");
            return 0;
        }
        intClassLayer = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Number of Hierarchical Class Layer: %d\n", intClassLayer);
        }
    }
    // parameters for vocabulary size
    intIndex = SearchArgs("-vocab-size", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Size of vocabulary not specified!\n");
            return 0;
        }
        intVocabSize = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Size of Vocabulary: %d\n", intVocabSize);
        }
    }
    // parameters for maximum length of sentence
    intIndex = SearchArgs("-max-len", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Maximum length of sentence not specified!\n");
            return 0;
        }
        intMaxLen = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Maximum length of sentence: %d\n", intMaxLen);
        }
    }
    // parameters for number of hidden layers
    intIndex = SearchArgs("-layer-num", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Number of hidden layers not specified!\n");
            return 0;
        }
        intLayerNum = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Number of Hidden Layers: %d\n", intLayerNum);
        }
    }
    // parameters for number of hidden layers
    intIndex = SearchArgs("-layer-names", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+intLayerNum+1 > argc)
        {
            printf("ERROR: Name of each hidden layer not specified!\n");
            return 0;
        }
        if(intLayerName != NULL) delete [] intLayerName;
        intLayerName = new int [intLayerNum];
        for(int i=0; i<intLayerNum; i++)
        {
            intLayerName[i] = atoi(argv[intIndex+i+1]);
        }
        if (bDebugMode > 0)
        {
            for(int i=0; i<intLayerNum; i++)
            {
                printf("Code for %d-th Hidden Layer's Name: %d\n", i+1, intLayerName[i]);
            }
        }
    }
    // parameters for number of hidden layers
    intIndex = SearchArgs("-layer-size", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+intLayerNum+1 > argc)
        {
            printf("ERROR: Size of each hidden layer not specified!\n");
            return 0;
        }
        if(intLayerSize != NULL) delete [] intLayerSize;
        intLayerSize = new int [intLayerNum];
        for(int i=0; i<intLayerNum; i++)
        {
            intLayerSize[i] = atoi(argv[intIndex+i+1]);
        }
        if (bDebugMode > 0)
        {
            for(int i=0; i<intLayerNum; i++)
            {
                printf("Size of %d-th Hidden Layer: %d\n", i+1, intLayerSize[i]);
            }
        }
    }
    // parameters for activation function in hidden layers
    intIndex = SearchArgs("-acfun", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Activation function not specified!\n");
            return 0;
        }
        intAcFun = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Activation Function: %d\n", intAcFun);
        }
    }
    // parameters for activation function of gates in LSTM
    intIndex = SearchArgs("-gatefun", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Activation function for gates not specified!\n");
            return 0;
        }
        intGateFun = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Activation Function for Gates: %d\n", intAcFun);
        }
    }
    // parameters for file type
    intIndex = SearchArgs("-file-type", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: File type not specified!\n");
            return 0;
        }
        intFileType = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("File Type: %d\n", intFileType);
        }
    }
    // parameters for file type
    intIndex = SearchArgs("-input-unit", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Input unit not specified!\n");
            return 0;
        }
        intInputUnit = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Input Unit: %d\n", intInputUnit);
        }
    }
    // parameters for maximum iteration
    intIndex = SearchArgs("-iter", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Maximum iteration not specified!\n");
            return 0;
        }
        intIterNum = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Maximum Iteration: %d\n", intIterNum);
        }
    }
    // parameters for random seed
    intIndex = SearchArgs("-seed", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Seed for random generator not specified!\n");
            return 0;
        }
        intRandomSeed = atoi(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Seed for Random Generator: %d\n", intRandomSeed);
        }
    }
    // parameters for bias terms
    intIndex = SearchArgs("-bias", argc, argv);
    if (intIndex > 0)
    {
        bEnBias = 1;
        if (bDebugMode > 0)
        {
            printf("Bias Terms: %d\n", bEnBias);
        }
    }
    // parameters for direct connections
    intIndex = SearchArgs("-direct", argc, argv);
    if (intIndex > 0)
    {
        bEnDirect = 1;
        if (bDebugMode > 0)
        {
            printf("Direct Connections: %d\n", bEnDirect);
        }
    }
    // parameters for learning rate
    intIndex = SearchArgs("-alpha", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Learning rate not specified!\n");
            return 0;
        }
        dAlpha = atof(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Learning Rate: %.5f\n", dAlpha);
        }
    }
    // parameters for regularization parameters
    intIndex = SearchArgs("-beta", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Regularization parameters not specified!\n");
            return 0;
        }
        dBeta = atof(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Regularization Parameters: %.9f\n", dBeta);
        }
    }
    // parameters for regularization parameters
    intIndex = SearchArgs("-mini-improv", argc, argv);
    if (intIndex > 0)
    {
        if (intIndex+1 == argc)
        {
            printf("ERROR: Minimun impovement not specified!\n");
            return 0;
        }
        dMinRate = atof(argv[intIndex+1]);
        if (bDebugMode > 0)
        {
            printf("Minimun Impovement on Validation Data: %.5f\n", dMinRate);
        }
    }
    // parameters for caching
    intIndex = SearchArgs("-cache", argc, argv);
    if (intIndex > 0)
    {
        bEnCache = 1;
        if (bDebugMode > 0)
        {
            printf("Enable caching: %d\n", bEnCache);
        }
    }
    // parameters for dynamic model
    intIndex = SearchArgs("-dynamic", argc, argv);
    if (intIndex > 0)
    {
        bEnDynamic = 1;
        if (bDebugMode > 0)
        {
            printf("Enable dynamic model: %d\n", bEnDynamic);
        }
    }
    // parameters for reversing word order
    intIndex = SearchArgs("-reverse", argc, argv);
    if (intIndex > 0)
    {
        bEnReverse = 1;
        if (bDebugMode > 0)
        {
            printf("Reverse word order: %d\n", bEnReverse);
        }
    }

    NNLM objNNLM;
    if(bEnTrainMode)
    {
        // enable training mode
        objNNLM.EnTrainMode(bEnTrainMode);
        // set the model name
        objNNLM.SetModelName(strModelName);
        // set training files
        objNNLM.SetTrainFiles(strTrainFiles);
        // set validation files
        objNNLM.SetValidFiles(strValidFiles);
        // set test files
        objNNLM.SetTestFiles(strTestFiles);
        // set the output path
        objNNLM.SetOutputPath(strOutputPath);
        // set the marks for seoeical words
        objNNLM.SetMarks(strStartMark, strEndMark, strUnknown);
        // set dimesion of feature vector
        objNNLM.SetVectorDim(intVectorDim);
        // set grammer order for feedforward NNLM
        objNNLM.SetGramOrder(intGramOrder);
        // set the number of word classes
        objNNLM.SetClassSize(intClassSize);
        // set the word class assignment algorithm
        objNNLM.SetClassAssign(intClassAssign);
        // set the number of hierarchical class layer
        objNNLM.SetClassLayer(intClassLayer);
        // set the size of vocabulary
        objNNLM.SetVocabSize(intVocabSize);
        // set activation function for hidden layers
        objNNLM.SetAcFun(intAcFun);
        // set activation function for gates in LSTM
        objNNLM.SetGateFun(intGateFun);
        // set file type
        objNNLM.SetFileType(intFileType);
        // set input unit
        objNNLM.SetInputUnit(intInputUnit);
        // set parameters for hidden layers
        objNNLM.SetHiddenLayers(intLayerNum, intLayerName, intLayerSize);
        // set maximum number of iteration
        objNNLM.SetIterNum(intIterNum);
        // set seed for random generator
        objNNLM.SetRandomSeed(intRandomSeed);
        // enable or disable bias terms
        objNNLM.EnBias(bEnBias);
        // enable or disable diect connections
        objNNLM.EnDirect(bEnDirect);
        // enable or disable caching
        objNNLM.EnCache(bEnCache);
        // reverse word order or not
        objNNLM.EnReverse(bEnReverse);
        // enable or disable dynamic model
        objNNLM.EnDynamic(bEnDynamic);
        // set learning rate
        objNNLM.SetAlpha(dAlpha);
        // set regularization parameters
        objNNLM.SetBeta(dBeta);
        // set the minimun of improvement rate on validation data
        objNNLM.SetMinRate(dMinRate);
        // set maximum length of sequence
        objNNLM.SetMaxLength(intMaxLen);
        // initialize the nodes
        objNNLM.InitModel();
        objNNLM.TrainModel();
        if(bEnTestMode)
        {
            objNNLM.TestModel();
        }
    }
    else
    {
        if(bEnTestMode)
        {
            // enable test mode
            objNNLM.EnTestMode(bEnTestMode);
            // set external model file
            objNNLM.SetModelFile(strModelFile);
            objNNLM.LoadModel();
            // enable or disable dynamic model
            objNNLM.EnDynamic(bEnDynamic);
            // set test files
            objNNLM.SetTestFiles(strTestFiles);
            objNNLM.TestModel();
        }
    }
    return 0;
}