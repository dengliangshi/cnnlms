#!/bin/bash
make clean
make
time ./bin/nnlm -name fnn_30_30 -train ./input/train/ -valid ./input/valid/ -test ./input/test/ -output ./output/ -order 5 -vector-dim 30 -class-size 100 -layer-num 1 -layer-names 0 -layer-size 30 -acfun 0 -alpha 0.01
time ./bin/nnlm -name rnn_30_30 -train ./input/train/ -valid ./input/valid/ -test ./input/test/ -output ./output/ -vector-dim 30 -class-size 100 -layer-num 1 -layer-names 1 -layer-size 30 -acfun 2 -alpha 0.1
time ./bin/nnlm -name lstm_30_30 -train ./input/train/ -valid ./input/valid/ -test ./input/test/ -output ./output/ -vector-dim 30 -class-size 100 -layer-num 1 -layer-names 2 -layer-size 30 -acfun 2 -gatefun 2 -alpha 0.1