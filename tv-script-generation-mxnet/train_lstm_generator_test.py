#!/usr/bin/python3

import unittest
import train_lstm_generator
import text_batcher
import mxnet as mx

batch_size = 6
sequence_length = 32

class TestTrainLSTMGen(unittest.TestCase):
    def setUp(self):
        self.batcher = text_batcher.TextBatcher('data/sample_text.txt')
        self.batcher.map_chars()
        self.batcher.make_batches(sequence_length)
        self.dataloader = mx.gluon.data.DataLoader(self.batcher, batch_size,True)
        self.net = train_lstm_generator.makeLSTMmodel(self.batcher.vocab_size)

    def test_forward_pass_LSTM(self):
        for (data,label) in self.dataloader:
            output = self.net(data)
            self.assertEqual(output.shape,label.shape)

    def test_train_model(self):
        train_lstm_generator.train(self.net,self.batcher,2,.001,1,1)