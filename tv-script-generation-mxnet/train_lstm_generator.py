#!/usr/bin/python3

import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd
import random
import text_batcher

mx.random.seed(1)
random.seed(1)

def train(net, dataset, batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
    # Adam.
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

    avg_loss = 50
    for epoch in range(1, epochs + 1):
        print("Batch size %d, Learning rate %f, Epoch %d" %
              (batch_size, trainer.learning_rate, epoch))
        if epoch > 1:
            print("average loss: %f" % avg_loss)
        for (data, label) in data_iter:
            with autograd.record():
                output = net(data)
                loss = average_ce_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            avg_loss = loss

def makeLSTMmodel(vocab_size,hidden_size=256, LSTM_layers=1):
    model = mx.gluon.nn.Sequential()
    with model.name_scope():
        model.add(mx.gluon.rnn.LSTM(hidden_size,num_layers=LSTM_layers))
        model.add(mx.gluon.nn.Dense(vocab_size, flatten=False))
    model.initialize()
    return model

def cross_entropy(yhat, y):
    return -1*nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))

def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)