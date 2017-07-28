// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "nnlmlib.h"

// --------------------------------------------------------Global Strings----------------------------------------------------


// ------------------------------------------------------------Main----------------------------------------------------------
NNLM::NNLM()
{
    strVersion = "1.0";
    bEnTrainMode = 0;
    bEnTestMode = 0;

    C = NULL;
    V = NULL;
    Vc = NULL;
    M = NULL;
    Mc = NULL;
    d = NULL;
    dc = NULL;

    Vb = NULL;
    Vcb = NULL;
    Mb = NULL;
    Mcb = NULL;

    x = NULL;
    y = NULL;
    c = NULL;
    s = NULL;
    wIndexs = NULL;
    intLayerName = NULL;
    intLayerSize = NULL;
}

NNLM::~NNLM()
{
    if(C != NULL) delete [] C;
    if(V != NULL) delete [] V;
    if(Vc != NULL) delete [] Vc;
    if(M != NULL) delete [] M;
    if(Mc != NULL) delete [] Mc;
    if(d != NULL) delete [] d;
    if(dc != NULL) delete [] dc;

    if(Vb != NULL) delete [] Vb;
    if(Vcb != NULL) delete [] Vcb;
    if(Mb != NULL) delete [] Mb;
    if(Mcb != NULL) delete [] Mcb;

    if(x != NULL) delete [] x;
    if(y != NULL) delete [] y;
    if(c != NULL) delete [] c;
    if(wIndexs != NULL) delete [] wIndexs;
}


double NNLM::Random(double dLower, double dUpper)
{
    return rand() / (double)RAND_MAX * (dUpper - dLower) + dLower;
}

void NNLM::SetMarks(string startMark, string endMark, string unKnown)
{
    strStartMark = startMark;
    strEndMark = endMark;
    strUnknown = unKnown;
}

void NNLM::SetHiddenLayers(int layerNum, int *layerName, int *layerSize)
{
    intLayerNum = layerNum;
    intLayerName = layerName;
    intLayerSize = layerSize;
}

void NNLM::InitModel()
{
    intLength = 0;
    srand(intRandomSeed);
    strModelFile = strOutputPath + strModelName;
    //build up vocabulary
    objVocab.SetClassSize(intClassSize);
    objVocab.SetClassAssign(intClassAssign);
    objVocab.SetClassLayer(intClassLayer);
    objVocab.SetVocabSize(intVocabSize);
    objVocab.SetInputUnit(intInputUnit);
    objVocab.SetMarks(strStartMark, strEndMark, strUnknown);
    objVocab.InitModel(strTrainFiles, intFileType);
    if(bEnTrainMode)
    {
        intVocabSize = objVocab.Generate();
        intTotalWords = objVocab.GetTotalWords();
    }
    intFirstLayer = intLayerName[0];
    if(intFirstLayer == FNN)
    {
        intInputSize = (intGramOrder - 1) * intVectorDim;
    }
    else
    {
        intInputSize = intVectorDim;
    }
    // initialize hidden layers
    objHidden.SetAcFun(intAcFun);
    objHidden.SetGateFun(intGateFun);
    objHidden.EnBias(bEnBias);
    objHidden.SetMaxLength(intMaxLen);
    objHidden.InitModel(intInputSize, intLayerNum,
        intLayerSize, intLayerName);
    intHiddenSize = intLayerSize[intLayerNum-1];
    
    intClassOutput = 0;
    // hierarchical word class
    if(intClassAssign==0 && intClassLayer>1)
    {
        for(int i=0; i<intClassLayer; i++)
        {
            intClassOutput = (intClassOutput + 1) * intClassSize;
        }
        bEnHierarchies = 1;
    }
    else
    {
        intClassOutput = intClassSize;
        bEnHierarchies = 0;
    }

    // allocate memeory for parameters of NNLM
    C = new double [intVocabSize*intVectorDim];
    V = new double [intVocabSize*intHiddenSize];
    Vc = new double [intClassOutput*intHiddenSize];
    M = new double [intVocabSize*intInputSize];
    Mc = new double [intClassOutput*intInputSize];
    d = new double [intVocabSize];
    dc = new double [intClassOutput];
    
    Vb = new double [intVocabSize*intHiddenSize];
    Vcb = new double [intClassOutput*intHiddenSize];
    Mb = new double [intVocabSize*intInputSize];
    Mcb = new double [intClassOutput*intInputSize];

    x = new weight [intMaxLen*intInputSize];
    y = new weight [intMaxLen*intVocabSize];
    c = new weight [intMaxLen*intClassOutput];
    wIndexs = new index [intMaxLen];

    if(bEnTrainMode)
    {
        for(int i=0; i<intVocabSize*intVectorDim; i++)
        {
            C[i] = Random(-sqrt(1.0/intVectorDim),
                    sqrt(1.0/intVectorDim));
        }
        for(int i=0; i<intVocabSize*intHiddenSize; i++)
        {
            V[i] = Random(-sqrt(1.0/intHiddenSize),
                sqrt(1.0/intHiddenSize));
        }
        for(int i=0; i<intClassOutput*intHiddenSize; i++)
        {
            Vc[i] = Random(-sqrt(1.0/intHiddenSize),
                sqrt(1.0/intHiddenSize));
        }
        if(bEnDirect)
        {
            for(int i=0; i<intVocabSize*intInputSize; i++)
            {
                M[i] = Random(-sqrt(1.0/intInputSize),
                    sqrt(1.0/intInputSize));
            }
            for(int i=0; i<intClassOutput*intInputSize; i++)
            {
                Mc[i] = Random(-sqrt(1.0/intInputSize),
                    sqrt(1.0/intInputSize));
            }
        }
        if(bEnBias)
        {
            for(int i=0; i<intVocabSize; i++)
            {
                d[i] = Random(-0.1, 0.1);
            }
            for(int i=0; i<intClassOutput; i++)
            {
                dc[i] = Random(-0.1, 0.1);
            }
        }
    
        // set the vector for unknown word to zeros
        objVocab.GetIndex(strUnknown, &unKnownIndex);
        for(int i=0; i<intVectorDim; i++)
        {
            C[intVectorDim*unKnownIndex.intIndex+i] = 0;
        }
    }
}

void NNLM::InitLog()
{
    FILE *fin;
    // output log
    strLogFile = strModelFile + ".log";
    fin = fopen(strLogFile.c_str(), "wb");
    fprintf(fin, "Parameters of language model:\n");
    fprintf(fin, "Toolkit version: V%s.\n", strVersion.c_str());
    fprintf(fin, "Training files: %s.\n", strTrainFiles.c_str());
    fprintf(fin, "Validation files: %s.\n", strValidFiles.c_str());
    fprintf(fin, "Test files: %s.\n", strTestFiles.c_str());
    fprintf(fin, "Mark for start of sentence: %s.\n", strStartMark.c_str());
    fprintf(fin, "Mark for end of sentence: %s.\n", strEndMark.c_str());
    fprintf(fin, "Mark for wprds out of vocabulary: %s.\n", strUnknown.c_str());
    fprintf(fin, "Dimension of feature vector: %d.\n", intVectorDim);
    fprintf(fin, "Grammer order: %d.\n", intGramOrder);
    fprintf(fin, "Size of input layer: %d.\n", intInputSize);
    fprintf(fin, "Number of word class: %d.\n", intClassSize);
    fprintf(fin, "Code for word class assignment: %d.\n", intClassAssign);
    fprintf(fin, "Number of hierarchical class layer: %d.\n", intClassLayer);
    fprintf(fin, "Size of vocabulary: %d.\n", intVocabSize);
    fprintf(fin, "Total words in training data: %d.\n", intTotalWords);
    fprintf(fin, "Size of last hidden layer: %d.\n", intHiddenSize);
    fprintf(fin, "Number of hidden layers: %d.\n", intLayerNum);
    for(int i=0; i<intLayerNum; i++)
    {
        fprintf(fin, "\t%dth hidden layer, name: %d, size: %d.\n",
            (i+1), intLayerName[i], intLayerSize[i]);
    }
    fprintf(fin, "Code for first hidden layer: %d.\n", intFirstLayer);
    fprintf(fin, "Code for activation function: %d.\n", intAcFun);
    fprintf(fin, "Input level of sequences: %d.\n", intInputUnit);
    fprintf(fin, "Maximum of iteraton: %d.\n", intIterNum);
    fprintf(fin, "Random seed: %d.\n", intRandomSeed);
    fprintf(fin, "Enable bias terms: %d.\n", bEnBias);
    fprintf(fin, "Enable direct connections: %d.\n", bEnDirect);
    fprintf(fin, "Enable caching: %d.\n", bEnCache);
    fprintf(fin, "Enable dynamic model: %d.\n", bEnDynamic);
    fprintf(fin, "Reverse word order: %d.\n", bEnReverse);
    fprintf(fin, "Learning rate: %.8f.\n", dAlpha);
    fprintf(fin, "Regularization parameters: %.8f.\n", dBeta);
    fprintf(fin, "Maximum of input sequnece's length: %d.\n", intMaxLen);
    fclose(fin);
}

void NNLM::Softmax(weight *x, int lowerIndex, int upperIndex)
{
    double reSum = 0;
    for(int i=lowerIndex; i<=upperIndex; i++)
    {
        reSum += exp(x[i].re);
    }
    for(int i=lowerIndex; i<=upperIndex; i++)
    {
        x[i].re =  exp(x[i].re) / reSum;
    }
}

void NNLM::Run()
{
    int intIndex;         // word index
    int intClass;         // class index
    int intOffset;
    int intWordIndex;
    int intLowerIndex;
    int intUpperIndex;
    double dClassProb = 1;

    for(int t=0; t<intLength; t++)
    {
        if(intFirstLayer == FNN)
        {
            intOffset = t - intGramOrder + 2;
            for(int i=0; i<intGramOrder-1; i++)
            {
                intIndex = intOffset + i;
                if(intIndex < 0) {intIndex = 0;}
                intWordIndex = wIndexs[intIndex].intIndex;
                for(int j=0; j<intVectorDim; j++)
                {
                    x[intInputSize*(t+1)+intVectorDim*i+j].re = C[intVectorDim*intWordIndex+j];
                    x[intInputSize*(t+1)+intVectorDim*i+j].er = 0;
                }
            }
        }
        else
        {
            intWordIndex = wIndexs[t].intIndex;
            for(int j=0; j<intVectorDim; j++)
            {
                x[intInputSize*(t+1)+j].re = C[intVectorDim*intWordIndex+j];
                x[intInputSize*(t+1)+j].er = 0;
            }
        }
    }
    // run hidden layer(s)
    s = objHidden.Run(x, intLength);
    for(int t=0; t<intLength; t++)
    {
        // probability distribution for words in the same class
        intIndex = wIndexs[t+1].intIndex;
        intClass = wIndexs[t+1].intClass;
        objVocab.GetRange(intClass, &intLowerIndex, &intUpperIndex);
        for(int i=intLowerIndex; i<=intUpperIndex; i++)
        {
            y[intVocabSize*t+i].re = 0;  
            for(int j=0; j<intHiddenSize; j++)
            {
                y[intVocabSize*t+i].re += V[intHiddenSize*i+j] * s[intHiddenSize*(t+1)+j].re;
            }
            if(bEnDirect)
            {
                for(int j=0; j<intInputSize; j++)
                {
                    y[intVocabSize*t+i].re += M[intInputSize*i+j] * x[intInputSize*(t+1)+j].re;
                }
            }
            if(bEnBias) {y[intVocabSize*t+i].re += d[i];}
        }
        Softmax(y, intVocabSize*t+intLowerIndex, intVocabSize*t+intUpperIndex);
        // probability distribution for word classes
        if(bEnHierarchies)
        {
            int intSumSize = 0;
            intOffset = 0;
            int classIndex;
            dClassProb = 1;

            for(int l=0; l<intClassLayer; l++)
            {
                for(int i=0; i<intClassSize; i++)
                {
                    classIndex = intOffset + intSumSize + i;
                    c[intClassOutput*t+classIndex].re = 0;
                    for(int j=0; j<intHiddenSize; j++)
                    {
                        c[intClassOutput*t+classIndex].re += Vc[intHiddenSize*classIndex+j] * s[intHiddenSize*(t+1)+j].re;
                    }
                    if(bEnDirect)
                    {
                        for(int j=0; j<intInputSize; j++)
                        {
                            c[intClassOutput*t+classIndex].re += Mc[intInputSize*classIndex+j] * x[intInputSize*(t+1)+j].re;
                        }
                    }
                    if(bEnBias) {c[intClassOutput*t+classIndex].re += dc[classIndex];}
                }
                intClass = wIndexs[t+1].intVector[l];
                Softmax(c, intClassOutput*t+intOffset+intSumSize, intClassOutput*t+intOffset+intSumSize+intClassSize-1);
                dClassProb = dClassProb*c[intClassOutput*t+intSumSize+intClass].re;
                intSumSize = (intSumSize + 1) * intClassSize;
                intOffset = intClass * intClassSize;
            }
        }
        else
        {
            for(int i=0; i<intClassSize; i++)
            {
                c[intClassSize*t+i].re = 0;
                for(int j=0; j<intHiddenSize; j++)
                {
                    c[intClassSize*t+i].re += Vc[intHiddenSize*i+j] * s[intHiddenSize*(t+1)+j].re;
                }
                if(bEnDirect)
                {
                    for(int j=0; j<intInputSize; j++)
                    {
                        c[intClassSize*t+i].re += Mc[intInputSize*i+j] * x[intInputSize*(t+1)+j].re;
                    }
                }
                if(bEnBias) {c[intClassSize*t+i].re += dc[i];}
            }
            Softmax(c, intClassSize*t, intClassSize*(t+1)-1);
            dClassProb = c[intClassSize*t+intClass].re;
        }
        if(intIndex == unKnownIndex.intIndex)
        {
            continue;
        }
        else
        {
            intWordNum ++;
            dLogP += log(y[intVocabSize*t+intIndex].re * dClassProb);
        }
    }
}

void NNLM::Update()
{
    double dLdp;
    double error;
    int classIndex;
    int intSumSize;
    int intClass;
    int intOffset;
    int intIndex = 0;
    int intLowerIndex;
    int intUpperIndex;
    int intWordIndex;

    double dRBeta = dAlpha*dBeta;
    for(int i=0; i<intVocabSize*intHiddenSize; i++) { Vb[i] = V[i]; }
    for(int i=0; i<intClassOutput*intHiddenSize; i++) { Vcb[i] = Vc[i]; }
    if(bEnDirect)
    {
        for(int i=0; i<intVocabSize*intInputSize; i++) { Mb[i] = M[i]; }
        for(int i=0; i<intClassOutput*intInputSize; i++) { Mcb[i] = Mc[i]; }
    }
    // calculate error gradient
    for(int t=0; t<intLength; t++)
    {
        intIndex = wIndexs[t+1].intIndex;
        intClass = wIndexs[t+1].intClass;
        // error of outputs for words
        objVocab.GetRange(intClass, &intLowerIndex, &intUpperIndex);
        for(int i=intLowerIndex; i<=intUpperIndex; i++)
        {
            y[intVocabSize*t+i].er = 0 - y[intVocabSize*t+i].re;
        }
        y[intVocabSize*t+intIndex].er = 1 - y[intVocabSize*t+intIndex].re;
        if(bEnHierarchies)
        {
            intSumSize = 0;
            intOffset = 0;
            for(int l=0; l<intClassLayer; l++)
            {
                for(int i=0; i<intClassSize; i++)
                {
                    classIndex = intOffset + intSumSize + i;
                    c[intClassOutput*t+classIndex].er = 0 - c[intClassOutput*t+classIndex].re;
                }
                intClass = wIndexs[t+1].intVector[l];
                c[intClassOutput*t+intSumSize+intClass].er = 1 - c[intClassOutput*t+intSumSize+intClass].re;
                intSumSize = (intSumSize + 1) * intClassSize;
                intOffset = intClass * intClassSize;
            }
        }
        else
        {
            // error of outputs for word classes
            for(int i=0; i<intClassSize; i++)
            {
                c[intClassSize*t+i].er = 0 - c[intClassSize*t+i].re;
            }
            c[intClassSize*t+intClass].er = 1 - c[intClassSize*t+intClass].re;

        }
        
        // error gradient for matrix V, M and hidden states
        for(int i=intLowerIndex; i<=intUpperIndex; i++)
        {
            dLdp = y[intVocabSize*t+i].er;
            for(int j=0; j<intHiddenSize; j++)
            {
                s[intHiddenSize*(t+1)+j].er += dLdp * Vb[intHiddenSize*i+j];
                error = dLdp * s[intHiddenSize*(t+1)+j].re;
                V[intHiddenSize*i+j] += dAlpha*error - dRBeta*Vb[intHiddenSize*i+j];
            }
            if(bEnDirect)
            {
                for(int j=0; j<intInputSize; j++)
                {
                    x[intInputSize*(t+1)+j].er += dLdp * Mb[intInputSize*i+j];
                    error = dLdp * x[intInputSize*(t+1)+j].re;
                    M[intInputSize*i+j] += dAlpha*error - dRBeta*Mb[intInputSize*i+j];
                }
            }
            if(bEnBias)
            {
                d[i] += dAlpha*dLdp - dRBeta*d[i];
            }
        }

        // error gradient for matrix Vc Mc and hidden states
        if(bEnHierarchies)
        {
            intSumSize = 0;
            intOffset = 0;
            for(int l=0; l<intClassLayer; l++)
            {
                for(int i=0; i<intClassSize; i++)
                {
                    classIndex = intOffset + intSumSize + i;
                    dLdp = c[intClassOutput*t+classIndex].er;
                    for(int j=0; j<intHiddenSize; j++)
                    {
                        s[intHiddenSize*(t+1)+j].er += dLdp * Vcb[intHiddenSize*classIndex+j];
                        error = dLdp * s[intHiddenSize*(t+1)+j].re;
                        Vc[intHiddenSize*classIndex+j] += dAlpha*error - dRBeta*Vcb[intHiddenSize*classIndex+j];
                    }
                    if(bEnDirect)
                    {
                        for(int j=0; j<intInputSize; j++)
                        {
                            x[intInputSize*(t+1)+j].er += dLdp * Mcb[intInputSize*classIndex+j];
                            error = dLdp * x[intInputSize*(t+1)+j].re;
                            Mc[intInputSize*classIndex+j] += dAlpha*error - dRBeta*Mcb[intInputSize*classIndex+j];
                        }
                    }
                    if(bEnBias)
                    {
                        dc[classIndex] += dAlpha*dLdp - dRBeta*dc[classIndex];
                    }
                }
                intClass = wIndexs[t+1].intVector[l];
                intSumSize = (intSumSize + 1) * intClassSize;
                intOffset = intClass * intClassSize;
            }
        }
        else
        {
            for(int i=0; i<intClassSize; i++)
            {
                dLdp = c[intClassSize*t+i].er;
                for(int j=0; j<intHiddenSize; j++)
                {
                    s[intHiddenSize*(t+1)+j].er += dLdp * Vcb[intHiddenSize*i+j];
                    error = dLdp * s[intHiddenSize*(t+1)+j].re;
                    Vc[intHiddenSize*i+j] += dAlpha*error - dRBeta*Vcb[intHiddenSize*i+j];
                    
                }
                if(bEnDirect)
                {
                    for(int j=0; j<intInputSize; j++)
                    {
                        x[intInputSize*(t+1)+j].er += dLdp * Mcb[intInputSize*i+j];
                        error = dLdp * x[intInputSize*(t+1)+j].re;
                        Mc[intInputSize*i+j] += dAlpha*error - dRBeta*Mcb[intInputSize*i+j];
                    }
                }
                if(bEnBias)
                {
                    dc[i] += dAlpha*dLdp - dRBeta*dc[i];
                }
            }
        }
            
    }

    objHidden.Update(dAlpha, dRBeta);
    for(int t=0; t<intLength; t++)
    {
        if(intFirstLayer == FNN)
        {
            intOffset = t - intGramOrder + 2;
            for(int i=0; i<intGramOrder-1; i++)
            {
                intIndex = intOffset + i;
                if(intIndex<0) {intIndex=0;}
                intWordIndex = wIndexs[intIndex].intIndex;
                for(int j=0; j<intVectorDim; j++)
                {
                    error = x[intInputSize*(t+1)+intVectorDim*i+j].er;
                    C[intVectorDim*intWordIndex+j] += dAlpha*error - dRBeta*C[intVectorDim*intWordIndex+j];
                }
            }
        }
        else
        {
            intWordIndex = wIndexs[t].intIndex;
            for(int j=0; j<intVectorDim; j++)
            {
                error = x[intInputSize*(t+1)+j].er;
                C[intVectorDim*intWordIndex+j] += dAlpha*error - dRBeta*C[intVectorDim*intWordIndex+j];
            }
        }
    }
}

void NNLM::ResetMaxLength()
{
    index *temIndex;
    temIndex = wIndexs;
    intMaxLen += 50;
    wIndexs = new index [intMaxLen];
    for(int i=0; i<intLength+1; i++)
    {
        wIndexs[i] = temIndex[i];
    }
    delete [] temIndex;
    if(x != NULL) delete [] x;
    if(y != NULL) delete [] y;
    if(c != NULL) delete [] c;
    x = new weight [intMaxLen*intInputSize];
    y = new weight [intMaxLen*intVocabSize];
    c = new weight [intMaxLen*intClassOutput];
    objHidden.ResetMaxLength(intMaxLen);
}

void NNLM::ReadSentence(FILE *fin)
{
    string strWord;
    index wIndex;
    intLength = 0;

    while(!feof(fin))
    {
        if(intInputUnit == 0)
        {
            strWord = objVocab.ReadWord(fin);
        }
        else
        {
            strWord = objVocab.ReadChar(fin);
        }
        objVocab.GetIndex(strWord, &wIndex);
        intLength ++;
        if(intLength > intMaxLen-1)
        {// doublelocate memeory for sequences
            ResetMaxLength();
        }
        wIndexs[intLength] = wIndex;
        // finish reading a sentence
        if(strWord.compare(strEndMark) == 0)
        { 
            break;
        }
    }
}

void NNLM::GetLocalTime(char* strTime)
{
    time_t timestamp;           // local time
    struct tm *t;               // time struct

    timestamp = time(NULL);
    t = localtime(&timestamp);
    sprintf(strTime, "%d-%02d-%02d %02d:%02d:%02d",
        1900 + t->tm_year, t->tm_mon+1,
        t->tm_mday, t->tm_hour,
        t->tm_min, t->tm_sec
    );
}

void NNLM::TrainModel()
{
    DIR *dp;                // handle of open directory
    FILE *fin;              // handle of open file
    FILE *finLog;           // handle of log file
    double dPreLogP;        // previous logarithm probabilty of validation data
    struct dirent *dirp;    // struct for file info
    struct stat statBuf;    // properties of directionary or file
    string strFileName;     // name of a train file
    string strFilePath;     // absolute path of a train file
    string strWord;         // a word from train files
    index wIndex;           // word index and class
    index wChange;          // change word order
    bool bAlphaCut = 0;     // if cut off learning rate
    int intPreWordNum;      // word count
    char strTime[20];       // local time string
    clock_t start, now;     // time stamp
    
    dPreLogP = -1e+10;
    InitLog();
    strLogFile = strModelFile + ".log";
    finLog = fopen(strLogFile.c_str(), "a");
    GetLocalTime(strTime);
    fprintf(finLog, "%s: training starts!\n", strTime);
    // start of a sentence
    objVocab.GetIndex(strStartMark, &wIndex);
    for(int i=0; i<intIterNum; i++)
    {
        intWordNum = 0;
        dLogP = 0;
        start = clock();
        intPreWordNum = 0;
        // traverse training files under given directory
        dp = opendir(strTrainFiles.c_str());
        while((dirp=readdir(dp))!=NULL)
        {
            string strFileName(dirp->d_name);
            strFilePath = strTrainFiles + strFileName;
            stat(strFilePath.c_str(), &statBuf);
            // ignore subdirectory
            if(S_ISDIR(statBuf.st_mode)) continue;
            // get word from a file
            fin = fopen(strFilePath.c_str(), "rb");
            while(!feof(fin))
            {
                if(bEnCache)
                {
                    // using caching
                    for(int j=0; j<intHiddenSize; j++)
                    {
                        s[j].re = s[intLength*intHiddenSize+j].re;
                    }
                }
                wIndexs[0] = wIndex;
                ReadSentence(fin);
                if(bEnReverse)
                {
                    // reverse word order
                    for(int t=0; t<(intLength+1)/2; t++)
                    {
                        wChange = wIndexs[t];
                        wIndexs[t] = wIndexs[intLength-t];
                        wIndexs[intLength-t] = wChange;
                    }
                }
                Run();
                Update();
                if((intWordNum - intPreWordNum) > 10000)
                {
                    now = clock();
                    printf("%c%dth interation, training entropy is %.2f, alpha is %.5f, %.2f words/s, progress: %.2f%%.", 13, (i+1),
                        -dLogP/log(2)/(double)intWordNum, dAlpha, intWordNum/(double)((now-start)/1000000.0), 100*intWordNum/(double)intTotalWords);
                    fflush(stdout);
                    intPreWordNum = intWordNum;
                }
            }
            fclose(fin);
            if(bEnCache)
            {
                for(int j=0; j<intHiddenSize; j++)
                {
                    s[j].re = 0.1;
                }
            }
        }
        closedir(dp);
        now = clock();
        printf("%c%dth interation, training entropy is %.2f, alpha is %.5f, %.2f words/s, ", 13, (i+1),
            -dLogP/log(2)/(double)intWordNum, dAlpha, intWordNum/(double)((now-start)/1000000.0));
        GetLocalTime(strTime);
        fprintf(finLog, "%s: %dth interation, training entropy is %.2f, alpha is %.5f, %.2f words/s, ", strTime,
            (i+1), -dLogP/log(2)/intWordNum, dAlpha, intWordNum/(double)((now-start)/1000000.0));
        // run model on validation data
        ValidModel(finLog);
        if(dLogP * dMinRate < dPreLogP)
        {
            if(bAlphaCut) break;
            bAlphaCut = 1;
        }
        dPreLogP = dLogP;
        if(bAlphaCut)
        {
            dAlpha = dAlpha / 2;
        }
    }
    SaveModel(finLog);
    fclose(finLog);
}

void NNLM::ValidModel(FILE *finLog)
{
    DIR *dp;                // handle of open directory
    FILE *fin;              // handle of open file
    struct dirent *dirp;    // struct for file info
    struct stat statBuf;    // properties of directionary or file
    string strFileName;     // name of a train file
    string strFilePath;     // absolute path of a train file
    string strWord;         // a word from train files
    index wIndex;           // word index and class
    index wChange;          // change word order

    dLogP = 0;
    intWordNum = 0;
    // start of a sentence
    objVocab.GetIndex(strStartMark, &wIndex);
    // traverse training files under given 
    dp = opendir(strValidFiles.c_str());
    while((dirp=readdir(dp))!=NULL)
    {
        string strFileName(dirp->d_name);
        strFilePath = strValidFiles + strFileName;
        stat(strFilePath.c_str(), &statBuf);
        // ignore subdirectory
        if(S_ISDIR(statBuf.st_mode)) continue;
        // get word from a file
        fin = fopen(strFilePath.c_str(), "rb");
        while(!feof(fin))
        {
            if(bEnCache)
            {
                for(int i=0; i<intHiddenSize; i++)
                {
                    s[i].re = s[intLength*intHiddenSize+i].re;
                }
            }
            wIndexs[0] = wIndex;
            ReadSentence(fin);
            if(bEnReverse)
            {// reverse word order
                for(int t=0; t<(intLength+1)/2; t++)
                {
                    wChange = wIndexs[t];
                    wIndexs[t] = wIndexs[intLength-t];
                    wIndexs[intLength-t] = wChange;
                }
            }
            Run();
        }
        fclose(fin);
        if(bEnCache)
        {
            for(int j=0; j<intHiddenSize; j++)
            {
                s[j].re = 0.1;
            }
        }
    }
    closedir(dp);
    printf("validation entropy is %.2f.\n", -dLogP/log(2)/intWordNum);
    fprintf(finLog, "validation entropy is %.2f.\n", -dLogP/log(2)/intWordNum);
}

void NNLM::TestModel()
{
    DIR *dp;                // handle of open directory
    FILE *fin;              // handle of open file
    FILE *finLog;           // handle of log file
    struct dirent *dirp;    // struct for file info
    struct stat statBuf;    // properties of directionary or file
    string strFileName;     // name of a train file
    string strFilePath;     // absolute path of a train file
    string strWord;         // a word from train files
    index wIndex;           // word index and class
    index wChange;          // change word order
    char strTime[20];       // local time string
    clock_t start, now;     // time stamp

    strLogFile = strModelFile + ".log";
    finLog = fopen(strLogFile.c_str(), "a");
    GetLocalTime(strTime);
    fprintf(finLog, "%s: test starts!\n", strTime);

    intWordNum = 0;
    dLogP = 0;
    start = clock();
    // words out of vocabulary
    objVocab.GetIndex(strUnknown, &unKnownIndex);
    // start of a sentence
    objVocab.GetIndex(strStartMark, &wIndex);
    // traverse training files under given directory
    dp = opendir(strTestFiles.c_str());
    while((dirp=readdir(dp))!=NULL)
    {
        string strFileName(dirp->d_name);
        strFilePath = strTestFiles + strFileName;
        stat(strFilePath.c_str(), &statBuf);
        // ignore subdirectory
        if(S_ISDIR(statBuf.st_mode)) continue;
        // get word from a file
        fin = fopen(strFilePath.c_str(), "rb");
        while(!feof(fin))
        {
            if(bEnCache)
            {
                for(int i=0; i<intHiddenSize; i++)
                {
                    s[i].re = s[intLength*intHiddenSize+i].re;
                }
            }
            wIndexs[0] = wIndex;
            ReadSentence(fin);
            if(bEnReverse)
            {// reverse word order
                for(int t=0; t<(intLength+1)/2; t++)
                {
                    wChange = wIndexs[t];
                    wIndexs[t] = wIndexs[intLength-t];
                    wIndexs[intLength-t] = wChange;
                }
            }
            Run();
            if(bEnDynamic)
            {
                Update();
            }
        }
        fclose(fin);
        if(bEnCache)
        {
            for(int j=0; j<intHiddenSize; j++)
            {
                s[j].re = 0.1;
            }
        }
    }
    closedir(dp);
    now = clock();
    printf("PPL of test data set %.2f, %.2f words/s.\n",
        pow(2, -dLogP/log(2)/intWordNum), intWordNum/(double)((now-start)/1000000.0));
    GetLocalTime(strTime);
    fprintf(finLog, "%s: PPL of test data set %.2f, %.2f words/s.\n", strTime,
        pow(2, -dLogP/log(2)/intWordNum), intWordNum/(double)((now-start)/1000000.0));
    fclose(finLog);
}

void NNLM::SaveModel(FILE *finLog)
{
    FILE *fout;             // handle of open file
    char strTime[20];       // local time string

    fout = fopen(strModelFile.c_str(), "wb");
    if (fout == NULL)
    {
        printf("Cannot create model file: %s\n", strModelFile.c_str());
        GetLocalTime(strTime);
        fprintf(finLog, "%s: failed to save model!\n", strTime);
    }
    else
    {
        fprintf(fout, "Version: %s\n", strVersion.c_str());
        fprintf(fout, "Model Name: %s\n", strModelName.c_str());
        fprintf(fout, "Training file(s): %s\n", strTrainFiles.c_str());
        fprintf(fout, "Validation file(s): %s\n", strValidFiles.c_str());
        fprintf(fout, "Test file(s): %s\n", strTestFiles.c_str());
        fprintf(fout, "Output path: %s\n", strOutputPath.c_str());
        fprintf(fout, "Log file: %s\n", strLogFile.c_str());
        fprintf(fout, "Mark for start of sentence: %s\n", strStartMark.c_str());
        fprintf(fout, "Mark for end of sentence: %s\n", strEndMark.c_str());
        fprintf(fout, "Mark for words out of vocabulary: %s\n", strUnknown.c_str());
        fprintf(fout, "Dimension of feature vector: %d\n", intVectorDim);
        fprintf(fout, "Grammer order: %d\n", intGramOrder);
        fprintf(fout, "Size of input layer: %d\n", intInputSize);
        fprintf(fout, "Number of word class: %d\n", intClassSize);
        fprintf(fout, "Code for word class assignment: %d\n", intClassAssign);
        fprintf(fout, "Number of hierarchical class layer: %d\n", intClassLayer);
        fprintf(fout, "Size of vocabulary: %d\n", intVocabSize);
        fprintf(fout, "Total words in training data: %d\n", intTotalWords);
        fprintf(fout, "Size of last hidden layer: %d\n", intHiddenSize);
        fprintf(fout, "Number of hidden layers: %d\n", intLayerNum);
        fprintf(fout, "Maximum length of sequences: %d\n", intMaxLen);
        fprintf(fout, "Code for first hidden layer: %d\n", intFirstLayer);
        for(int i=0; i<intLayerNum; i++)
        {
            fprintf(fout, "%dth hidden layer name: %d\n", (i+1), intLayerName[i]);
            fprintf(fout, "%dth hidden layer size: %d\n", (i+1), intLayerSize[i]);
        }
        fprintf(fout, "Code for activation function: %d\n", intAcFun);
        fprintf(fout, "Code for gate function: %d\n", intGateFun);
        fprintf(fout, "File type: %d\n", intFileType);
        fprintf(fout, "Input level of sequences: %d\n", intInputUnit);
        fprintf(fout, "Maximum of iteraton: %d\n", intIterNum);
        fprintf(fout, "Random seed: %d\n", intRandomSeed);
        fprintf(fout, "Enable bias terms: %d\n", bEnBias);
        fprintf(fout, "Enable direct connections: %d\n", bEnDirect);
        fprintf(fout, "Enable caching: %d\n", bEnCache);
        fprintf(fout, "Enable dynamic model: %d\n", bEnDynamic);
        fprintf(fout, "Reverse word order: %d\n", bEnReverse);
        fprintf(fout, "Learning rate: %.10f\n", dAlpha);
        fprintf(fout, "Regularization parameters: %.10f.\n", dBeta);
        fprintf(fout, "Minimum improvement on validation data: %.10f\n", dMinRate);
        fprintf(fout, "Projection matrix C:\n");
        for(int i=0; i<intVocabSize*intVectorDim; i++)
        {
            fprintf(fout, "%lf\n", C[i]);
        }
        fprintf(fout, "Weight matrix V:\n");
        for(int i=0; i<intVocabSize*intHiddenSize; i++)
        {
            fprintf(fout, "%lf\n", V[i]);
        }
        fprintf(fout, "Weight matrix Vc:\n");
        for(int i=0; i<intClassOutput*intHiddenSize; i++)
        {
            fprintf(fout, "%lf\n", Vc[i]);
        }
        fprintf(fout, "Weight matrix M:\n");
        for(int i=0; i<intVocabSize*intInputSize; i++)
        {
            fprintf(fout, "%lf\n", M[i]);
        }
        fprintf(fout, "Weight matrix Mc:\n");
        for(int i=0; i<intClassOutput*intInputSize; i++)
        {
            fprintf(fout, "%lf\n", Mc[i]);
        }
        fprintf(fout, "Bisa terms d:\n");
        for(int i=0; i<intVocabSize; i++)
        {
            fprintf(fout, "%lf\n", d[i]);
        }
        fprintf(fout, "Bisa terms dc:\n");
        for(int i=0; i<intClassOutput; i++)
        {
            fprintf(fout, "%lf\n", dc[i]);
        }
        objVocab.SaveModel(fout);
        objHidden.SaveModel(fout);
        GetLocalTime(strTime);
        fprintf(finLog, "%s: model is saved sucessfully!\n", strTime);
    }
    fclose(fout);
}

void NNLM::GotoDelimiter(int intDelimiter, FILE *fin)
{
    int ch = 0;
    while(ch != intDelimiter)
    {
        ch = fgetc(fin);
        if(feof(fin))
        {
            printf("Unexpected end of file!\n");
            exit(1);
        }
    }
}

void NNLM::LoadModel()
{
    FILE *fin;              // handle of open file
    FILE *finLog;           // handle of log file
    char strVar[1000];      // for string varibles
    int intVar;             // for integer or bool varibles
    char strTime[20];       // local time string

    fin = fopen(strModelFile.c_str(), "rb");
    if (fin == NULL)
    {
        printf("Cannot open model file: %s\n", strModelFile.c_str());
    }
    else
    {
        // version
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        if((string)strVar != strVersion)
        {
            printf("Model file is not generated by current version toolkit!\n");
            exit(1);
        }
        // model name
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strModelName = strVar;
        // training file(s)
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strTrainFiles = strVar;
        // validation file(s)
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strValidFiles = strVar;
        // test file(s)
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strTestFiles = strVar;
        // output path
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strOutputPath = strVar;
        // log file
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strLogFile = strVar;
        // mark for start of sentence
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strStartMark = strVar;
        // mark for end of sentence
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strEndMark = strVar;
        // mark for words out of vocabulary
        GotoDelimiter(':', fin);
        fscanf(fin, "%s", strVar);
        strUnknown = strVar;
        // dimension of feature vector
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intVectorDim);
        // grammer order
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intGramOrder);
        // size of input layer
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intInputSize);
        // number of word class
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intClassSize);
        // code for word class assignment
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intClassAssign);
        // number of hierarchical word class layer
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intClassLayer);
        // size of vocabulary
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intVocabSize);
        // total words in training data
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intTotalWords);
        // size of last hidden layer
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intHiddenSize);
        // number of hidden layers
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intLayerNum);
        // maximum length of sequences
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intMaxLen);
        // code for first hidden layer
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intFirstLayer);
        if(intLayerName == NULL)
        {
            intLayerName = new int[intLayerNum];
        }
        if(intLayerSize == NULL)
        {
            intLayerSize = new int[intLayerNum];
        }
        // name and size of each hidden layer
        for(int i=0; i<intLayerNum; i++)
        { 
            //hidden layer name
            GotoDelimiter(':', fin);
            fscanf(fin, "%d", &intVar);
            intLayerName[i] = intVar;
            // hidden layer size
            GotoDelimiter(':', fin);
            fscanf(fin, "%d", &intVar);
            intLayerSize[i] = intVar;
        }
        // code for activation function
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intAcFun);
        // code for gate function
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intGateFun);
        // file type
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intFileType);
        // input level of sequences
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intInputUnit);
        // maximum of iteraton
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intIterNum);
        // random seed
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intRandomSeed);
        // enable bias terms
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intVar);
        bEnBias = intVar;
        // enable direct connections
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intVar);
        bEnDirect = intVar;
        // enable caching
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intVar);
        bEnCache = intVar;
        // enable dynamic model
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intVar);
        bEnDynamic = intVar;
        // reverse word order
        GotoDelimiter(':', fin);
        fscanf(fin, "%d", &intVar);
        bEnReverse = intVar;
        // learning rate
        GotoDelimiter(':', fin);
        fscanf(fin, "%lf", &dAlpha);
        // regularization parameters
        GotoDelimiter(':', fin);
        fscanf(fin, "%lf", &dBeta);
        // minimum improvement on validation data
        GotoDelimiter(':', fin);
        fscanf(fin, "%lf", &dMinRate);
        InitModel();
        // projection matric C
        GotoDelimiter(':', fin);
        for(int i=0; i<intVocabSize*intVectorDim; i++)
        {
            fscanf(fin, "%lf", &C[i]);
        }
        // weight matric V
        GotoDelimiter(':', fin);
        for(int i=0; i<intVocabSize*intHiddenSize; i++)
        {
            fscanf(fin, "%lf", &V[i]);
        }
        // weight matric Vc
        GotoDelimiter(':', fin);
        for(int i=0; i<intClassOutput*intHiddenSize; i++)
        {
            fscanf(fin, "%lf", &Vc[i]);
        }
        // weight matric M
        GotoDelimiter(':', fin);
        for(int i=0; i<intVocabSize*intInputSize; i++)
        {
            fscanf(fin, "%lf", &M[i]);
        }
        // weight matric Mc
        GotoDelimiter(':', fin);
        for(int i=0; i<intClassOutput*intInputSize; i++)
        {
            fscanf(fin, "%lf", &Mc[i]);
        }
        // bias term d
        GotoDelimiter(':', fin);
        for(int i=0; i<intVocabSize; i++)
        {
            fscanf(fin, "%lf", &d[i]);
        }
        // bias term dc
        GotoDelimiter(':', fin);
        for(int i=0; i<intClassOutput; i++)
        {
            fscanf(fin, "%lf", &dc[i]);
        }
        objVocab.LoadModel(fin);
        objHidden.LoadModel(fin);
        strLogFile = strModelFile + ".log";
        finLog = fopen(strLogFile.c_str(), "a");
        GetLocalTime(strTime);
        fprintf(finLog, "%s: model is reloaded from %s sucessfully!\n",
            strTime, strModelFile.c_str());
        fclose(finLog);
    }
    fclose(fin);
}