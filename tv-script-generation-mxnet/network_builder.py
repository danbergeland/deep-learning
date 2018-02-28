#!/usr/bin/python3

import mxnet as mx

def makeLSTMmodel(vocab_size, embedded_features=64,hidden_size=256, LSTM_layers=1):
    model = mx.gluon.nn.Sequential()
    with model.name_scope():
        model.add(mx.gluon.nn.Embedding(vocab_size, embedded_features))
        model.add(mx.gluon.rnn.LSTM(hidden_size,num_layers=LSTM_layers))
        model.add(mx.gluon.nn.Dense(vocab_size, flatten=False))
    model.initialize()
    return model