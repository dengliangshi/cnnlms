## Neural Network Language Models

This is a Neural Networl Language Models (NNLMs) toolkit which supports Feed-forward Neural Network (FNN), Recurrent Neural Network (RNN), Long Short Term Memory (LSTM) RNN, Bidirectional RNN and Bidirectional LSTM. Neural network language models with multible hidden layers also can be built with this toolkit, and the architecture of hidden layers can be different.

## Configuration
Options and arguments for NNLMs are as follows:

>-acfun

Specify the code for activation function in hidden layer(s), like `-acfun 0`, and default is 0. Only one kind of activation function can be specified for all hidden layers. The codes of activation function are as following: 0 - tanh, 1 - hard tanh, 2 - sigmoid, 3 - hard sigmoid and 4 - relu.

>-alpha

Learning rate for stochastic gradient descent (SGD) algorithm, like `-alpha 0.1`, and default is 0.01.

>-beta

Regularization paramters for stochastic gradient descent (SGD) algorithm, like `-beta 1.0e-3`, and default is 1.0e-3.

>-bias

Enable the bias terms in neural network. It is an option, and no value should be assigned, just like `-bias`. Without this option, no bias terms will be used.

>-cache

By this option, in each article, the information from all previous context will be taken into account to predict next word using inner state vectors. This option does not work for FNNLM.

>-class-assign

Code for word classes assignment algorithm, like `-class-assign 2`, default is 2. The codes for algorithms are as followings: 0 - assign word classes uniformly, 1 - assign word classes according to words's frequency, 2 - assign word classes according to words' square frequency.

>-class-layer

For hierarchical neural network language model, specify the number of layers, like `-class-layer 3`, and default is 1. This arguement only works when the code of word classes assignment algorithm is set as 0.

>-class-size

When using word classes to speed up neural network language model, the number of word classes should be given, like `-class-size 100`, and default is 100. Word classes can be disabled by setting the number of word classes to be 1.

>-debug

Enable debug mode.

>-direct

Enable the direct connections between input layer and output layer. It is an option, and no value is needed, like `-direct`. Without this option, no direct connections will be enabled.

>-dynamic

If this option is given, model will be update during test phrase.

>-end-mark

Mark for the end of sentence, like `-end-mark </s>`, default is `</s>`.

>-gatefun

The code of activation function for the gates in long short term memory (LSTM) recurrent neural network. The codes are the same as the ones of activation function in hidden layer, and are listed here for convenience: 0 - tanh, 1 - hard tanh, 2 - sigmoid, 3 - hard sigmoid and 4 - relu.

>-help

Get the help document of this toolkit, and it is an optionm no value is needed. 

>-input-unit

Specify the unit into which word sequence will be split, like '-input-unit 0', and default is 0. Only 0 or 1 should be given, and 0 stands for words while 1 for characters.

>-iter

The maximum number of iteration for training, like `-iter 10`, and default is 20. Training will be terminated when the number of iteration reach this maximum.

>-layer-num

Specify the number of hidden layers, like `-layer-num 1`, and default is 1.

>-layer-names

The code for each hidden layers, like `-layer-names 2 1`, and default is 0. The number of values for this argument should be equal to the number of hidden layers. The codes for neural networks are as following:  0 -FNN, 1 - RNN, 2 - LSTM, 3 - BiRNN and 4 - BiLSTM.

>-layer-size

The size of each hidden layer, like `-layer-size 20 20`, and default is 100. The number of value for this argument should equal to the number of hidden layers, and the value should be match with each other. When BiRNN or BiSTM is used, the size of this hidden layer should be an even number, because the size of forward layer and backward one will be set as half of the given value.

>-max-len

The possible maximum length of sentences in target data set, like `-max-len 100`, and default is 100. It is ok to have a guess for this argument, because this value will be enlarged if the length of a sentence exceeds the given one.

>-mini-improv

Minimum improvement rate of entropy on validation data, specified like `-mini-improv 1.003`, and default is 1.003. The learning rate will be cut by a ratio when the imporvement of entropy on validation data is less than this minimum rate the first time, and training will be terminated the second time. This is a early stop strategy to avoid overfitting.

>-model

Specify the path of model file, like `-model ./output/fnn_100_100`, and the whole model will be reloaded from this file.

>-name

Specify a name for your model, like `-name rnn_100_100`, which will be used as the name of output files.

>-order

N-gram order, for example `-order 5`. Only needed when the first hidden layer is FNN, and default is 5.

>-output

The directionary for output files, like `-output ./output/`. It is a required argument.

>-reverse

Reverse the order of words in each sentence.

>-seed

Seed for random generator, like `-seed 1`, and default is 1.

>-start-mark

Mark for the start of sentence, like `-start-mark <s>`, default is `<s>`.

>-test

The path of test file(s), like `-test ./input/test/`, and all the files under this given path will be taken as test files. it is optional for training but required for testing. If both training files and test files are given, test will be performed at once after training.

>-train

The path of training file(s), like `-train ./input/train/`, and all the files under this given path will be taken as training files. It is required for traning purpose.

>-unknown

Mark for words out of vocabulary, like `-unknown OOV`, and deafult is OOV.

>-valid

The path of validation file(s), like `-valid ./input/valid/`, and all the files under this given path will be taken as validation files. It is required for traning purpose.

>-vector-dim

Dimension of feature vectors for words, like `-vector-dim 100`, and default is 100.

>-vocab-size

Specify the maximum number of words in vocabuary, like `-vocab-size 100000`, and default is 1000,000. If the number words from training data exceeds given maximum, the words with low frequecy will be excluded from vocabulary. Otherwise, all words will be added into vacabulary, and the size of vocabulary will be reset as the number of words.

## Usage
The examples are given with this toolkit, please refer to the file `example.sh`. More details about the languagel models built in this toolkit, please refer to [my posts](https://dengliangshi.github.io/).

## License
The module is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).