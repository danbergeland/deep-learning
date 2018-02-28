#!/usr/bin/python3

import unittest
import train_lstm_generator
import text_batcher

batch_size = 6
sequence_length = 32

class TestTrainLSTMGen(unittest.TestCase):
    def setUp(self):
        self.batcher = text_batcher.TextBatcher('data/sample_text.txt')
        self.batcher.map_chars()
        self.batcher.make_batches(sequence_length)
        self.net = train_lstm_generator.makeLSTMmodel(self.batcher.vocab_size)

    def test_train(self):
        train_lstm_generator.train(self.net,self.batcher,batch_size,.001,1,batch_size*2)
