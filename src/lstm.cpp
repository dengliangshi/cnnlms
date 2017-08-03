// ---------------------------------------------------------Libraries--------------------------------------------------------
// Standard Library


// Third-party Libraries


// User Define Module
#include "lstm.h"


// --------------------------------------------------------Global Strings----------------------------------------------------
// i(t) = G(Ui*x(t) + Wi*s(t-1) + Vi*c(t-1) + bi)
// f(t) = G(Uf*x(t) + Wf*s(t-1) + Vf*c(t-1) + bf)
// g(t) = F(U*x(t) + W*s(t-1) + b)
// c(t) = f(t)*c(t-1) + i(t)*g(t)
// o(t) = G(Uo*x(t) + Wo*s(t-1) + Vo*c(t) + bo)
// h(t) = F(c(t))
// s(t) = o(t)*h(t)

// ------------------------------------------------------------Main----------------------------------------------------------
void LSTMNN::InitModel(int inputSize, int hiddenSize)
{
    bActive = 1;
    intLength = 0;
    intInputSize = inputSize;
    intHiddenSize = hiddenSize;

    s = new weight [intMaxLen * intHiddenSize];
    c = new weight [intMaxLen * intHiddenSize];
    g = new double [intMaxLen * intHiddenSize];
    h = new double [intMaxLen * intHiddenSize];
    ig = new double [intMaxLen * intHiddenSize];
    fg = new double [intMaxLen * intHiddenSize];
    og = new double [intMaxLen * intHiddenSize];
    for(int i=0; i<intHiddenSize; i++)
    {
        c[i].re = 0.1; c[i].er = 0;
        s[i].re = 0.1; s[i].er = 0;
    }
    iGate.InitModel(intInputSize, intHiddenSize, bEnBias);
    fGate.InitModel(intInputSize, intHiddenSize, bEnBias);
    oGate.InitModel(intInputSize, intHiddenSize, bEnBias);
    objNodes.InitModel(intInputSize, intHiddenSize, bEnBias);
}

void LSTMNN::ResetMaxLength(int maxLength)
{
    if(intMaxLen < maxLength)
    {
        intMaxLen = maxLength;
        if(s != NULL) delete [] s;
        if(c != NULL) delete [] c;
        if(h != NULL) delete [] h;
        if(g != NULL) delete [] g;
        if(ig != NULL) delete [] ig;
        if(fg != NULL) delete [] fg;
        if(og != NULL) delete [] og;
        s = new weight [intMaxLen * intHiddenSize];
        c = new weight [intMaxLen * intHiddenSize];
        g = new double [intMaxLen * intHiddenSize];
        h = new double [intMaxLen * intHiddenSize];
        ig = new double [intMaxLen * intHiddenSize];
        fg = new double [intMaxLen * intHiddenSize];
        og = new double [intMaxLen * intHiddenSize];
        for(int i=0; i<intHiddenSize; i++)
        { 
            c[i].re = 0.1; c[i].er = 0;
            s[i].re = 0.1; s[i].er = 0;
        }
    }
}


weight* LSTMNN::Run(weight *input, int length)
{
    int intIndex;
    int intNext;

    x = input;
    intLength = length;
    for(int t=0; t<intLength; t++)
    {
        for(int i=0; i<intHiddenSize; i++)
        {
            intIndex = intHiddenSize*t + i;
            intNext = intHiddenSize*(t+1) + i;
            ig[intIndex] = 0;
            fg[intIndex] = 0;
            g[intIndex] = 0;
            c[intNext].er = 0;
            
            for(int j=0; j<intInputSize; j++)
            {
                ig[intIndex] += iGate.U[intInputSize*i+j].re * x[intInputSize*(t+1)+j].re;
                fg[intIndex] += fGate.U[intInputSize*i+j].re * x[intInputSize*(t+1)+j].re;
                og[intIndex] += oGate.U[intInputSize*i+j].re * x[intInputSize*(t+1)+j].re;
                g[intIndex] += objNodes.U[intInputSize*i+j].re * x[intInputSize*(t+1)+j].re;
            }
            for(int j=0; j<intHiddenSize; j++)
            {
                ig[intIndex] += (iGate.W[intHiddenSize*i+j].re * s[intHiddenSize*t+j].re
                    + iGate.V[intHiddenSize*i+j].re * c[intHiddenSize*t+j].re);
                fg[intIndex] += (fGate.W[intHiddenSize*i+j].re * s[intHiddenSize*t+j].re
                    + fGate.V[intHiddenSize*i+j].re * c[intHiddenSize*t+j].re);
                g[intIndex] += objNodes.W[intHiddenSize*i+j].re * s[intHiddenSize*t+j].re;
            }
            if(bEnBias)
            {
                ig[intIndex] += iGate.b[i].re;
                fg[intIndex] += fGate.b[i].re;
                og[intIndex] += oGate.b[i].re;
                g[intIndex] += objNodes.b[i].re;
            }
            if(ig[intIndex] > 50) { ig[intIndex] = 50; }
            if(ig[intIndex] < -50) { ig[intIndex] = -50; }
            ig[intIndex] = AcFun(ig[intIndex], intGateFun);

            if(fg[intIndex] > 50) { fg[intIndex] = 50; }
            if(fg[intIndex] < -50) { fg[intIndex] = -50; }
            fg[intIndex] = AcFun(fg[intIndex], intGateFun);

            if(g[intIndex] > 50) { g[intIndex] = 50; }
            if(g[intIndex] < -50) { g[intIndex] = -50; }
            g[intIndex] = AcFun(g[intIndex], intAcFun);
            c[intNext].re = fg[intIndex]*c[intIndex].re + ig[intIndex]*g[intIndex];

            if(c[intNext].re > 50) { c[intNext].re = 50; }
            if(c[intNext].re < -50) { c[intNext].re = -50; }
            h[intIndex] = AcFun(c[intNext].re, intAcFun);
        }

        for(int i=0; i<intHiddenSize; i++)
        {
            intIndex = intHiddenSize*t + i;
            intNext = intHiddenSize*(t+1) + i;
            s[intNext].er = 0;
            for(int j=0; j<intHiddenSize; j++)
            {
                og[intIndex] += (oGate.W[intHiddenSize*i+j].re * s[intHiddenSize*t+j].re
                    + oGate.V[intHiddenSize*i+j].re * c[intHiddenSize*(t+1)+j].re);
            }
            if(og[intIndex] > 50) { og[intIndex] = 50; }
            if(og[intIndex] < -50) { og[intIndex] = -50; }
            og[intIndex] = AcFun(og[intIndex], intGateFun);
            
            s[intNext].re = og[intIndex]*h[intIndex];
        }
    }
    return s;
}

void LSTMNN::Update(double dAlpha, double dBeta)
{
    double dLdi;
    double dLdf;
    double dLdo;
    double dLdg;
    int intIndex;
    int intNext;

    for(int t=intLength-1; t>=0; t--)
    {
        for(int i=0; i<intHiddenSize; i++)
        {
            intIndex = intHiddenSize*t+i;
            intNext = intHiddenSize*(t+1)+i;

            dLdo = s[intNext].er * h[intIndex] * dAcFun(og[intIndex], intGateFun);
            for(int j=0; j<intHiddenSize; j++)
            {
                oGate.W[intHiddenSize*i+j].er += dLdo * s[intHiddenSize*t+j].re;
                s[intHiddenSize*t+j].er += dLdo * oGate.W[intHiddenSize*i+j].re;
                oGate.V[intHiddenSize*i+j].er += dLdo * c[intHiddenSize*(t+1)+j].re;
                c[intHiddenSize*(t+1)+j].er += dLdo * oGate.V[intHiddenSize*i+j].re;
            }
        }
        for(int i=0; i<intHiddenSize; i++)
        {
            intIndex = intHiddenSize*t+i;
            intNext = intHiddenSize*(t+1)+i;
            
            dLdo = s[intNext].er * h[intIndex] * dAcFun(og[intIndex], intGateFun);
            c[intNext].er += s[intNext].er * og[intIndex] * dAcFun(h[intIndex], intAcFun);
            dLdi = c[intNext].er * g[intIndex] * dAcFun(ig[intIndex], intGateFun);
            dLdf = c[intNext].er * c[intIndex].re * dAcFun(fg[intIndex], intGateFun);
            dLdg = c[intNext].er * ig[intIndex] * dAcFun(g[intIndex], intAcFun);
            c[intIndex].er += c[intNext].er * fg[intIndex];
            for(int j=0; j<intInputSize; j++)
            {
                x[intInputSize*(t+1)+j].er += dLdi * iGate.U[intInputSize*i+j].re;
                iGate.U[intInputSize*i+j].er += dLdi * x[intInputSize*(t+1)+j].re;
                x[intInputSize*(t+1)+j].er += dLdf * fGate.U[intInputSize*i+j].re;
                fGate.U[intInputSize*i+j].er += dLdf * x[intInputSize*(t+1)+j].re;
                x[intInputSize*(t+1)+j].er += dLdo * oGate.U[intInputSize*i+j].re;
                oGate.U[intInputSize*i+j].er += dLdo * x[intInputSize*(t+1)+j].re;
                x[intInputSize*(t+1)+j].er += dLdg * objNodes.U[intInputSize*i+j].re;
                objNodes.U[intInputSize*i+j].er += dLdg * x[intInputSize*(t+1)+j].re;
            }
            for(int j=0; j<intHiddenSize; j++)
            {
                iGate.W[intHiddenSize*i+j].er += dLdi * s[intHiddenSize*t+j].re;
                s[intHiddenSize*t+j].er += dLdi * iGate.W[intHiddenSize*i+j].re;
                iGate.V[intHiddenSize*i+j].er += dLdi * c[intHiddenSize*t+j].re;
                c[intHiddenSize*t+j].er += dLdi * iGate.V[intHiddenSize*i+j].re;

                fGate.W[intHiddenSize*i+j].er += dLdf * s[intHiddenSize*t+j].re;
                s[intHiddenSize*t+j].er += dLdf * fGate.W[intHiddenSize*i+j].re;
                fGate.V[intHiddenSize*i+j].er += dLdf * c[intHiddenSize*t+j].re;
                c[intHiddenSize*t+j].er += dLdf * fGate.V[intHiddenSize*i+j].re;

                objNodes.W[intHiddenSize*i+j].er += dLdg * s[intHiddenSize*t+j].re;
                s[intHiddenSize*t+j].er += dLdg * objNodes.W[intHiddenSize*i+j].re;
            }
            if(bEnBias)
            {
                iGate.b[i].er += dLdi;
                fGate.b[i].er += dLdf;
                oGate.b[i].er += dLdo;
                objNodes.b[i].er += dLdg;
            }
        }
    }
    iGate.Update(dAlpha, dBeta);
    fGate.Update(dAlpha, dBeta);
    oGate.Update(dAlpha, dBeta);
    objNodes.Update(dAlpha, dBeta);
}