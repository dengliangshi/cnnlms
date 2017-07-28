// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library
#include <stdlib.h>

// Third-party Libraries


// User Define Module
#include "nodes.h"

// --------------------------------------------------------Global Strings----------------------------------------------------


// ------------------------------------------------------------Main----------------------------------------------------------
Nodes::Nodes()
{
    U = NULL;
    W = NULL;
    V = NULL;
    b = NULL;
}

Nodes::~Nodes()
{
    if(U != NULL) delete [] U;
    if(W != NULL) delete [] W;
    if(V != NULL) delete [] V;
    if(b != NULL) delete [] b;
}

void Nodes::InitModel(int inputSize, int hiddenSize, bool enBias)
{
    intInputSize = inputSize;
    intHiddenSize = hiddenSize;
    bEnBias = enBias;

    // allocate memory to weight matrixes and bias vector
    U = new weight [intHiddenSize * intInputSize];
    W = new weight [intHiddenSize * intHiddenSize];
    V = new weight [intHiddenSize * intHiddenSize];
    b = new weight [intHiddenSize];

    // assign weight matrixes and bias vector with initialization value
    for(int i=0; i<intHiddenSize*intInputSize; i++)
    {
        U[i].re = Random(-sqrt(1.0/intInputSize),
                sqrt(1.0/intInputSize));
        U[i].er = 0;
    }
    for(int i=0; i<intHiddenSize*intHiddenSize; i++)
    {
        W[i].re = Random(-sqrt(1.0/intHiddenSize),
            sqrt(1.0/intHiddenSize));
        W[i].er = 0;
        V[i].re = Random(-sqrt(1.0/intHiddenSize),
            sqrt(1.0/intHiddenSize));
        V[i].er = 0;
    }
    if(bEnBias)
    {
        for(int i=0; i<intHiddenSize; i++)
        {
            b[i].re = Random(-0.1, 0.1);
            b[i].er = 0;
        }
    }
}

void Nodes::Update(double dAlpha, double dBeta)
{
    for(int i=0; i<intHiddenSize*intInputSize; i++)
    {
        U[i].re += dAlpha * U[i].er - dBeta * U[i].re;
        U[i].er = 0;
    }
    for(int i=0; i<intHiddenSize*intHiddenSize; i++)
    {
        W[i].re += dAlpha * W[i].er - dBeta * W[i].re;
        W[i].er = 0;
        V[i].re += dAlpha * V[i].er - dBeta * V[i].re;
        V[i].er = 0;
    }
    if(bEnBias)
    {
        for(int i=0; i<intHiddenSize; i++)
        {
            b[i].re += dAlpha * b[i].er - dBeta * b[i].re;
            b[i].er = 0;
        }
    }
}

double Nodes::Random(double dLower, double dUpper)
{
    return rand() / (double)RAND_MAX * (dUpper - dLower) + dLower;
}

void Nodes::SaveModel(FILE *fout)
{
    fprintf(fout, "Weight matrix U:\n");
    for(int i=0; i<intHiddenSize*intInputSize; i++)
    {
        fprintf(fout, "%.10f\n", U[i].re);
    }
    fprintf(fout, "Weight matrix W:\n");
    for(int i=0; i<intHiddenSize*intHiddenSize; i++)
    {
        fprintf(fout, "%.10f\n", W[i].re);
    }
    fprintf(fout, "Weight matrix V:\n");
    for(int i=0; i<intHiddenSize*intHiddenSize; i++)
    {
        fprintf(fout, "%.10f\n", V[i].re);
    }
    fprintf(fout, "Bias terms b:\n");
    for(int i=0; i<intHiddenSize; i++)
    {
        fprintf(fout, "%.10f\n", b[i].re);
    }
}

void Nodes::LoadModel(FILE *fin)
{
    // weight matrix U
    GotoDelimiter(':', fin);
    for(int i=0; i<intHiddenSize*intInputSize; i++)
    {
        fscanf(fin, "%lf", &U[i].re);
    }
    // weight matrix W
    GotoDelimiter(':', fin);
    for(int i=0; i<intHiddenSize*intHiddenSize; i++)
    {
        fscanf(fin, "%lf", &W[i].re);
    }
    // weight matrix V
    GotoDelimiter(':', fin);
    for(int i=0; i<intHiddenSize*intHiddenSize; i++)
    {
        fscanf(fin, "%lf", &V[i].re);
    }
    // bias term b
    GotoDelimiter(':', fin);
    for(int i=0; i<intHiddenSize; i++)
    {
        fscanf(fin, "%lf", &b[i].re);
    }
}

void Nodes::GotoDelimiter(int intDelimiter, FILE *fin)
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