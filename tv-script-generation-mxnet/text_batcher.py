#!/usr/bin/python3

import string
import json
import mxnet as mx
from mxnet import nd

punctuation_translations = {
    '.' : ' PERIOD ',
    ',' : ' COMMA ',
    '!' : ' EXCLAMATION ',
    '(' : ' LEFTPAREN ',
    ')' : ' RIGHTPAREN ',
    '?' : ' QUESTIONMARK ',
    ':' : ' COLON ',
    ';' : ' SEMICOLON ',
}

punctuation_textify = {
    'PERIOD' : '.',
    'COMMA' : ',',
    'EXCLAMATION' : '!',
    'LEFTPAREN' : '(',
    'RIGHTPAREN' : ')',
    'QUESTIONMARK' : '?',
    'COLON' : ':',
    'SEMICOLON' : ';',
}

class TextBatcher(mx.gluon.data.Dataset):
    def __init__(self, data_path, ctx=mx.cpu(0)):
        self.data_path = data_path
        self.vocab_to_num = {}
        self.num_to_vocab = {}
        self.full_text = ''
        self.chars = []
        self.words = []
        self.vocab_size = 0
        self.batches = []
        self.labels = []
        self.ctx = ctx

    def __getitem__(self, key):
        """Required override for gluon.dataset"""
        try:
            return (self.batches[key], self.labels[key])
        except IndexError:
            return IndexError

    def __len__(self):
        """Required ovverid for gluon.dataset"""
        return len(self.batches)

    def load_text(self):
        """loads text from connected .txt file"""
        with open(self.data_path) as source_file:
            self.full_text = source_file.read()

    def file_to_word_list(self):
        """loads and creates a unique set of words"""
        self.load_text()
        modified_text = self.full_text.translate(str.maketrans(punctuation_translations))
        self.words = set(modified_text.split())
        return self.words

    def file_to_char_list(self):
        """loads and creates a unique set of characters"""
        self.load_text()
        self.chars = set(list(self.full_text))
        return self.chars

    def map_words(self):
        """Creates dictionaries for converting words to numeric integers"""
        self.file_to_word_list()
        if self.words is not []:
            self.vocab_to_num = {c:int(i) for i,c in enumerate(self.words)}
            self.num_to_vocab = {int(self.vocab_to_num[vocab]):vocab for vocab in self.vocab_to_num}
            self.vocab_size = len(self.vocab_to_num)

    def map_chars(self):
        """Creates dictionaries for converting individual chars to integers"""
        self.file_to_char_list()
        if self.chars is not []:
            self.vocab_to_num = {c:int(i) for i,c in enumerate(self.chars)}
            self.num_to_vocab = {int(self.vocab_to_num[vocab]):vocab for vocab in self.vocab_to_num}
            self.vocab_size = len(self.vocab_to_num)
    
    def save_vocab_map(self, map_path):
        """Creates JSON file for map (will save either word or char maps, 
            based on the last use of map_chars or map_words)"""
        with open(map_path,'w') as output_file:
            json.dump(self.vocab_to_num, output_file)

    def load_vocab_map(self, map_path):
        """Loads saved JSON map (will load either chars or words to the 
            conversion dictionaries based on the save file)"""
        with open(map_path) as input_file:
            self.vocab_to_num = json.load(input_file)
            self.num_to_vocab = {int(self.vocab_to_num[vocab]):vocab for vocab in self.vocab_to_num}
            self.vocab_size = len(self.vocab_to_num)

    def convert_vocab_to_numeric(self, input_text=None, word_embedding=False):
        """For a text input, return a list of conversions to integers"""
        if input_text is None:
            input_text = self.full_text
        if word_embedding is True:
            input_text = input_text.translate(str.maketrans(punctuation_translations)).split()
        return  [self.vocab_to_num[character] for character in input_text]

    def convert_numeric_to_vocab(self, input_numbers):
        """For a list of integers, returns a list of vocab conversions"""
        return [self.num_to_vocab[num] for num in input_numbers]

    def one_hots(self, numerical_list):
        """Creates an ndarray of len(numerical_list) rows by vocab size columns with 1-hot embeddings"""
        result = nd.zeros((len(numerical_list),self.vocab_size), self.ctx)
        for i, num in enumerate(numerical_list):
            result[i,num] = 1.0
        return result

    def textify(self, one_hot_NDArray, word_embedding=False):
        """Returns a string from a one hot encoded array"""
        result = ''
        onehot_index = nd.argmax(one_hot_NDArray, axis=1).asnumpy()
        for char_index in onehot_index:
            next_vocab = self.num_to_vocab[char_index]
            if word_embedding:
                if next_vocab in punctuation_textify:
                    result += punctuation_textify[next_vocab]
                    continue
                result += ' ' + next_vocab
                continue
            result += self.num_to_vocab[char_index]
        return result

    def make_batches(self, sequence_length, word_embedding=False):
        """Batches the .txt file at this.data_path as one-hot (stored in this.batches)
            and creates one-hot labels (stored in this.labels)
            The indices of batches and labels are matched."""
        batches = []
        labels = []
        numeric_list=self.convert_vocab_to_numeric(word_embedding=word_embedding)
        batch_count = len(numeric_list)//sequence_length
        for i in range(batch_count):
            sequence = numeric_list[i*sequence_length:((i+1)*sequence_length)]
            label_text = numeric_list[i*sequence_length+1:((i+1)*sequence_length+1)]
            if(len(sequence)==len(label_text)):
                batches.append(self.one_hots(sequence))
                labels.append(self.one_hots(label_text))
        self.batches = batches
        self.labels = labels