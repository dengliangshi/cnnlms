// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "vocab.h"


// --------------------------------------------------------Global Strings----------------------------------------------------


// ------------------------------------------------------------Main----------------------------------------------------------
Vocab::Vocab()
{
    intSubClassSize = NULL;
}
Vocab::~Vocab()
{
    dictVocab.clear();
    if(intSubClassSize != NULL) delete [] intSubClassSize;
}

void Vocab::SetMarks(string startMark, string endMark, string unKnown)
{
    strStartMark = startMark;
    strEndMark = endMark;
    strUnknown = unKnown;
}

void Vocab::InitModel(string trainFiles, int fileType)
{
    intWordCount = 0;
    intTotalWords = 0;
    strTrainFiles = trainFiles;
    intFileType = fileType;
    if(intClassAssign==0 && intClassLayer > 1)
    {
        intBottomSize = pow(intClassSize, intClassLayer);
        bEnHierarchies = 1;
    }
    else
    {
        intBottomSize = intClassSize;
        bEnHierarchies = 0;
    }
    intSubClassSize = new int [intBottomSize];
}

int Vocab::Generate()
{
    DIR *dp;                // handle of open directory
    FILE *fin;              // handle of open file
    struct dirent *dirp;    // struct for file info
    struct stat statBuf;    // properties of directionary or file
    string strFileName;     // name of a train file
    string strFilePath;     // absolute path of a train file
    string strWord;         // a word from train files

    // failed to open directory of training files
    if((dp=opendir(strTrainFiles.c_str())) == NULL)
    {
        printf("Failed to open training files!\n");
        return 0;
    }
    // traverse training files under given directory
    while((dirp=readdir(dp))!=NULL)
    {
        string strFileName(dirp->d_name);
        strFilePath = strTrainFiles + strFileName;
        stat(strFilePath.c_str(), &statBuf);
        // ignore subdirectory
        if(S_ISDIR(statBuf.st_mode)) continue;
        AddWord(strStartMark);
        // get word from a file
        fin = fopen(strFilePath.c_str(), "rb");
        while(!feof(fin))
        {
            if(intInputUnit == 0)
            {
                strWord = ReadWord(fin);
            }
            else
            {
                strWord = ReadChar(fin);
            }
            AddWord(strWord);
            // add start mark of a sentence
            if(strWord.compare(strEndMark) ==0)
            {
                AddWord(strStartMark);
            }
        }
        fclose(fin);
    }
    closedir(dp);
    // sort words in vocabulary according to their frequency
    SortVocab();

    return intVocabSize;
}

string Vocab::ReadWord(FILE *fin)
{
    int intChar;            // ASSIC of the character
    string strWord;         // word string

    while(!feof(fin))
    {
        intChar = fgetc(fin);
        // ignore the return character
        if (intChar == 13) continue;
        // the separator of words
        if((intChar == ' ') || (intChar == '\t') || (intChar == '\n'))
        {
            // finish reading a word
            if(!strWord.empty())
            {
                // the last word of a sentence
                if(intChar == '\n') ungetc(intChar, fin);
                break;
            }
            // the end of a sentence
            if(intChar == '\n')
            {
                return strEndMark;
            }
            else
            {
                continue;
            }
        }
        strWord += (char)intChar;
    }
    return strWord;
}

string Vocab::ReadChar(FILE *fin)
{
    int intChar;            // ASSIC of the character
    string strWord;         // word string

    while(!feof(fin))
    {
        intChar = fgetc(fin);
        // ignore the return character
        if (intChar == 13) continue;
        strWord += (char)intChar;
    }
    return strWord;
}

void Vocab::AddWord(string strWord)
{
    if(dictVocab.count(strWord) == 0)
    {
        word newWord;
        newWord.intFreq = 1;
        newWord.strName = strWord;
        dictVocab[strWord] = newWord;
        intWordCount ++;
    }
    else
    {
        dictVocab[strWord].intFreq += 1;
    }
    intTotalWords ++;
}

void Vocab::SortVocab()
{
    int intCount = 0;     // count the number of word
    int intMaxIndex;      // mark the index of word with highest frequency
    word wordTemp;        // temporary variable for swaping words
    word *wordList;       // order list of words
    dict::iterator t;     // iterator over vocabulary
    
    wordList = new word [intWordCount+1];
    // restore words into an order list
    for(t=dictVocab.begin(); t!=dictVocab.end(); t++)
    {
        wordList[intCount] = t->second;
        intCount ++;
    }
    // add unkown word into vocabulary
    if(intWordCount < intVocabSize - 1)
    {
        intVocabSize = intWordCount + 1;
        wordList[intVocabSize-1].strName = strUnknown;
        wordList[intVocabSize-1].intFreq = 1;
    }
    else
    {
        int intSumFreq = 0;
        // turncate the low frequency words if number of words exceeds specified vocabulary's size
        for(int i=intVocabSize-1; i<intWordCount; i++)
        {
            intSumFreq += wordList[i].intFreq;
        }
        wordList[intVocabSize-1].strName = strUnknown;
        wordList[intVocabSize-1].intFreq = intSumFreq;
    }
    // sort words in descent order according to their frequency
    for(int i = 0; i < intVocabSize; i++)
    {
        intMaxIndex = i;
        for(int j = i + 1; j < intVocabSize; j++)
        {
            if(wordList[j].intFreq > wordList[intMaxIndex].intFreq)
            {
                intMaxIndex = j;
            }
        }
        wordTemp = wordList[i];
        wordList[i] = wordList[intMaxIndex];
        wordList[intMaxIndex] = wordTemp;
    }
    // assign each word in vocabulary with a class
    AssignIndex(wordList);
    // add words into vocabulary and assign each word with index
    dictVocab.clear();
    for(int i = 0; i < intVocabSize; i++)
    {
        dictVocab[wordList[i].strName] = wordList[i];
    }
    delete [] wordList;
}

void Vocab::AssignIndex(word *wordList)
{
    int subClassSize = 0;   // word number of each class
    int intClassIndex = 0;  // index of class
    int intSumFreq = 0;     // summary of all words' frequency
    double dSqrtProb = 0;   // summary of the square of all words' probabilities
    double dProb = 0;       // the accumulation of current word's square probabilities

    switch(intClassAssign)
    {
        case 0:
        {
            if(intClassLayer > 1)
            {
                int intWordNum; 
                int *intClassVector;

                intClassVector = new int [intClassLayer];
                for(int i=0; i<intClassLayer; i++)
                {
                    intClassVector[i] = 0;
                }
                subClassSize = floor(intVocabSize/(pow(intClassSize, intClassLayer)));
                if(subClassSize == 0)
                {
                    printf("The size or layer number of word class is too large!\n");
                    exit(1);
                }
                for(int i=0; i<intVocabSize; i++)
                {
                    wordList[i].intIndex = i;
                    wordList[i].intVector = new int [intClassLayer];
                    if(i % subClassSize == 0 && i>0)
                    {
                        intClassIndex = intClassVector[intClassLayer-1];
                        intSubClassSize[intClassIndex] = i - 1;
                    }
                    for(int j=0; j<intClassLayer; j++)
                    {
                        intWordNum = subClassSize * pow(intClassSize, intClassLayer-j-1);
                        if(i % intWordNum == 0 && i>0)
                        {
                            if(intClassVector[j] < pow(intClassSize, j+1)-1)
                            {
                                intClassVector[j] ++;
                            }
                        }
                        wordList[i].intVector[j] = intClassVector[j];
                    }
                    wordList[i].intClass = intClassVector[intClassLayer-1];
                }
                intClassIndex = intClassVector[intClassLayer-1];
                intSubClassSize[intClassIndex] = intVocabSize - 1;
            }
            else
            {
                subClassSize = floor(intVocabSize/intClassSize);
                for(int i=0; i<intVocabSize; i++)
                {
                    wordList[i].intIndex = i;
                    if((i % subClassSize) == 0 && intClassIndex < intClassSize-1)
                    {
                        intSubClassSize[intClassIndex] = i-1;
                        intClassIndex ++;
                    }
                    wordList[i].intClass = intClassIndex;
                }
                intSubClassSize[intClassIndex] = intVocabSize - 1;
            }
            break;
        }
        case 1:
        {
            // caculate the summary of all words' frequency
            for(int i=0; i<intVocabSize; i++)
            {
                intSumFreq += wordList[i].intFreq;
            }
            // assgin class index
            for(int i=0; i<intVocabSize; i++)
            {
                wordList[i].intIndex = i;
                wordList[i].intClass = intClassIndex;
                dProb += wordList[i].intFreq/(double)intSumFreq;
                if((dProb > ((intClassIndex+1) / (double)intClassSize)) && (intClassIndex < intClassSize-1))
                {
                    intSubClassSize[intClassIndex] = wordList[i].intIndex;
                    intClassIndex ++;
                }
            }
            intSubClassSize[intClassIndex] = intVocabSize - 1;
            break;
        }
        case 2:
        {
            // caculate the summary of all words' frequency
            for(int i=0; i<intVocabSize; i++)
            {
                intSumFreq += wordList[i].intFreq;
            }
            // summary of the squre of all words' probabilities
            for(int i=0; i<intVocabSize; i++)
            {
                dSqrtProb += sqrt(wordList[i].intFreq/(double)intSumFreq);
            }
            // assgin class index
            for(int i=0; i<intVocabSize; i++)
            {
                wordList[i].intIndex = i;
                wordList[i].intClass = intClassIndex;
                dProb += sqrt(wordList[i].intFreq/(double)intSumFreq)/dSqrtProb;
                if((dProb > ((intClassIndex+1) / (double)intClassSize)) && (intClassIndex < intClassSize-1))
                {
                    intSubClassSize[intClassIndex] = wordList[i].intIndex;
                    intClassIndex ++;
                }
            }
            intSubClassSize[intClassIndex] = intVocabSize - 1;
            break;
        }
        default:
        {
            // caculate the summary of all words' frequency
            for(int i=0; i<intVocabSize; i++)
            {
                intSumFreq += wordList[i].intFreq;
            }
            // summary of the squre of all words' probabilities
            for(int i=0; i<intVocabSize; i++)
            {
                dSqrtProb += sqrt(wordList[i].intFreq/(double)intSumFreq);
            }
            // assgin class index
            for(int i=0; i<intVocabSize; i++)
            {
                wordList[i].intIndex = i;
                wordList[i].intClass = intClassIndex;
                dProb += sqrt(wordList[i].intFreq/(double)intSumFreq)/dSqrtProb;
                if((dProb > ((intClassIndex+1) / (double)intClassSize)) && (intClassIndex < intClassSize-1))
                {
                    intSubClassSize[intClassIndex] = wordList[i].intIndex;
                    intClassIndex ++;
                }
            }
            intSubClassSize[intClassIndex] = intVocabSize - 1;
            break;
        }

    }
}

void Vocab::GetIndex(string strWord, index *iWord)
{
    if(dictVocab.count(strWord) == 0)
    {
        iWord->intIndex = dictVocab.find(strUnknown)->second.intIndex;
        iWord->intClass = dictVocab.find(strUnknown)->second.intClass;
        iWord->intVector = dictVocab.find(strUnknown)->second.intVector;
    }
    else
    {
        iWord->intIndex = dictVocab.find(strWord)->second.intIndex;
        iWord->intClass = dictVocab.find(strWord)->second.intClass;
        iWord->intVector = dictVocab.find(strWord)->second.intVector;
    }
}

void Vocab::GetRange(int classIndex, int *lowerIndex, int *upperIndex)
{
    if(classIndex == 0)
    {
        *lowerIndex = 0;
    }
    else
    {
        *lowerIndex = intSubClassSize[classIndex-1] + 1;
    }
    *upperIndex = intSubClassSize[classIndex];
}

void Vocab::SaveModel(FILE *fout)
{
    dict::iterator t;     // iterator over vocabulary

    fprintf(fout, "Word count:%d\n", intWordCount);
    fprintf(fout, "Total words:%d\n", intTotalWords);
    fprintf(fout, "Dictionary:\n");
    for(t=dictVocab.begin(); t!=dictVocab.end(); t++)
    {
        fprintf(fout, "%s\n", t->second.strName.c_str());
        fprintf(fout, "%d\n", t->second.intIndex);
        fprintf(fout, "%d\n", t->second.intClass);
        fprintf(fout, "%d\n", t->second.intFreq);
        if(bEnHierarchies)
        {
            for(int i=0; i<intClassLayer; i++)
            {
                fprintf(fout, "%d\n", t->second.intVector[i]);
            }
        }
    }
    fprintf(fout, "Sub-class size:\n");
    for(int i=0; i<intBottomSize; i++)
    {
        fprintf(fout, "%d\n", intSubClassSize[i]);
    }
}

void Vocab::GotoDelimiter(int intDelimiter, FILE *fin)
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

void Vocab::LoadModel(FILE *fin)
{
    word wWord;             // word
    int intVar;             // for integer or bool varibles
    char strVar[1000];      // for string varibles
    dict::iterator t;       // iterator over vocabulary
    
    // word count
    GotoDelimiter(':', fin);
    fscanf(fin, "%d", &intWordCount);
    // total word number
    GotoDelimiter(':', fin);
    fscanf(fin, "%d", &intTotalWords);
    GotoDelimiter(':', fin);
    // dictionary
    for(int i=0; i<intVocabSize; i++)
    {
        fscanf(fin, "%s", strVar);
        wWord.strName = strVar;
        fscanf(fin, "%d", &intVar);
        wWord.intIndex = intVar;
        fscanf(fin, "%d", &intVar);
        wWord.intClass = intVar;
        fscanf(fin, "%d", &intVar);
        wWord.intFreq = intVar;
        if(bEnHierarchies)
        {
            wWord.intVector = new int [intClassLayer];
            for(int i=0; i<intClassLayer; i++)
            {
                fscanf(fin, "%d", &intVar);
                wWord.intVector[i] = intVar;
            }
        }
        dictVocab[strVar] = wWord;
    }
    GotoDelimiter(':', fin);
    for(int i=0; i<intBottomSize; i++)
    {
        fscanf(fin, "%d", &intVar);
        intSubClassSize[i] = intVar;
    }
}