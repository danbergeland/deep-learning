#!/usr/bin/python3

import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import ndarray as nd
import numpy as np
import random
import text_batcher
import time

mx.random.seed(1)
random.seed(1)

loss_func = mx.gluon.loss.SoftmaxCELoss(sparse_label=False)

def train(net, dataset, batch_size, lr, epochs):
    net.collect_params().initialize(mx.init.Normal(sigma=1), force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

    avg_loss = 50.
    for epoch in range(epochs):
        print("Batch size %d, Learning rate %f, Epoch %d" %
              (batch_size, trainer.learning_rate, epoch))
        print("average loss: ", avg_loss)
        for (data, label) in data_iter:
            with autograd.record():
                output = net(data)
                loss = loss_func(output, label)
            loss.backward()
            trainer.step(batch_size)
            avg_loss = nd.sum(loss)/batch_size
        if epoch%10 == 0:
            savepath = "checkpoints/moes_gen_epoch_"+str(epoch)+"_"+str(time.ctime()).replace(' ','_')
            net.collect_params().save(savepath)
        print(generate(net,dataset, 'Moe',128))

def makeLSTMmodel(vocab_size, hidden_size=256, LSTM_layers=1):
    model = mx.gluon.nn.Sequential()
    with model.name_scope():
        model.add(mx.gluon.rnn.LSTM(hidden_size,num_layers=LSTM_layers))
        model.add(mx.gluon.nn.Dense(vocab_size, flatten=False))
    model.initialize()
    return model

def generate(model, text_batcher, seed_text='', sequence_length=32):
    assert(len(seed_text)<sequence_length)
    sentence = nd.zeros((1,sequence_length,text_batcher.vocab_size))
    start_index = len(seed_text)
    for step in range(start_index, sequence_length):
        for i, symbol in enumerate(seed_text):
            sentence[0][i][text_batcher.vocab_to_num[symbol]] = 1
        output = nd.softmax(model(sentence)).asnumpy()
        #random sample from sentence[0][step]
        next_char_index = np.random.choice(text_batcher.vocab_size,p=output[0][step])
        next_char = text_batcher.convert_numeric_to_vocab([next_char_index])
        #add resulting character to seed_text
        seed_text+= next_char[0]
    return seed_text

if __name__ == '__main__':
    sequence_length = 256
    batch_size = 32
    learning_rate = .001
    epochs = 1000
    batcher = text_batcher.TextBatcher('data/moes_tavern_lines.txt')
    batcher.map_chars()
    batcher.save_vocab_map('data/moes_text_gen.json')
    batcher.make_batches(sequence_length, one_hot=True)
    print("batcher stuff",batcher)
    #dataloader = mx.gluon.data.DataLoader(batcher, batch_size,True)
    net = makeLSTMmodel(batcher.vocab_size)
    train(net,batcher,batch_size,learning_rate,epochs)
