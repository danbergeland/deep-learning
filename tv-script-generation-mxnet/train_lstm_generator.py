#!/usr/bin/python3

import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd
import numpy as np
import random
import text_batcher
import network_builder

mx.random.seed(1)
random.seed(1)

square_loss = gluon.loss.L2Loss()

def train(net, dataset, batch_size, lr, epochs, period):
    assert period >= batch_size and period % batch_size == 0
    # net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
    # # Adam.
    # trainer = gluon.Trainer(net.collect_params(), 'adam',
    #                         {'learning_rate': lr})
    # data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
    # total_loss = [np.mean(square_loss(net(X), y).asnumpy())]

    # for epoch in range(1, epochs + 1):
    #     for batch_i, (data, label) in enumerate(data_iter):
    #         with autograd.record():
    #             output = net(data)
    #             loss = square_loss(output, label)
    #         loss.backward()
    #         trainer.step(batch_size)

    #         if batch_i * batch_size % period == 0:
    #             total_loss.append(np.mean(square_loss(net(X), y).asnumpy()))
    #     print("Batch size %d, Learning rate %f, Epoch %d, loss %.4e" %
    #           (batch_size, trainer.learning_rate, epoch, total_loss[-1]))

    # print('w:', np.reshape(net[0].weight.data().asnumpy(), (1, -1)),
    #       'b:', net[0].bias.data().asnumpy()[0], '\n')
    # x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)