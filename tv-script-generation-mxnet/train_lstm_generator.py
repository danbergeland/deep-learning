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
        for (data, label) in data_iter:
            with autograd.record():
                output = net(data)
                loss = loss_func(output, label)
            loss.backward()
            trainer.step(batch_size)
            avg_loss = nd.sum(loss)/batch_size
        print("average loss: ", avg_loss)
        if epoch%10 == 0:
            savepath = "checkpoints/moes_gen_epoch_"+str(epoch)+"_"+str(time.ctime()).replace(' ','_')
            net.collect_params().save(savepath)
            print(generate(net,dataset,True, '',512))

def makeLSTMmodel(vocab_size, hidden_size=256, LSTM_layers=1):
    model = mx.gluon.nn.Sequential()
    with model.name_scope():
        model.add(mx.gluon.rnn.LSTM(hidden_size,num_layers=LSTM_layers))
        model.add(mx.gluon.nn.Dense(vocab_size, flatten=False))
    model.initialize()
    return model

def generate(model, text_batcher,greedy=False, seed_text='', sequence_length=32):
    assert(len(seed_text)<sequence_length)
    #load 1 sequence_length of real text to get the internal states correct.
    start_index = random.randint(0, len(text_batcher.full_text) - sequence_length - 1)
    seed_text = text_batcher.full_text[start_index:start_index+sequence_length]
    generated= ''
    for step in range(sequence_length):
        model_input = nd.zeros((1,sequence_length,text_batcher.vocab_size))
        for i, symbol in enumerate(seed_text):
            model_input[0][i][text_batcher.vocab_to_num[symbol]] = 1
        output = nd.softmax(model(model_input)).asnumpy()
        #random sample from sentence[0][step]
        next_char_index = None
        if greedy:
            next_char_index = np.argmax(output[0][sequence_length-1])
        else:
            next_char_index = np.random.choice(text_batcher.vocab_size,p=output[0][sequence_length-1])
        next_char = text_batcher.convert_numeric_to_vocab([next_char_index])
        #add resulting character to seed_text, remove first char.
        seed_text+= next_char[0]
        seed_text = seed_text[1:]
        generated+= next_char[0]
    return generated

if __name__ == '__main__':
    sequence_length = 40
    batch_size = 1024
    learning_rate = .0003
    epochs = 2000
    batcher = text_batcher.TextBatcher('data/moes_tavern_lines.txt')
    batcher.map_chars()
    batcher.save_vocab_map('data/moes_text_gen.json')
    batcher.make_batches(sequence_length, one_hot=True)
    net = makeLSTMmodel(batcher.vocab_size,512,3)
    train(net,batcher,batch_size,learning_rate,epochs)
