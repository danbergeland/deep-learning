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
        self.batcher.make_batches(sequence_length, one_hot=True)
        self.dataloader = mx.gluon.data.DataLoader(self.batcher, batch_size,True)
        self.net = train_lstm_generator.makeLSTMmodel(self.batcher.vocab_size)

    def test_forward_pass_LSTM_matches_label_shape(self):
        for (data,label) in self.dataloader:
            output = self.net(data)
            self.assertEqual(output.shape,label.shape)

    def test_train_model(self):
        #No assert because calling train should not result in errors, this test is to check for errors
        train_lstm_generator.train(self.net,self.batcher,batch_size,.001,1)

    def test_sample(self):
        generated = train_lstm_generator.generate(self.net,self.batcher,'test_string')
        self.assertEqual(len(generated),sequence_length)
