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
}

ctx = mx.cpu(0)

class DataHelper():
    def __init__(self, data_path):
        self.data_path = data_path
        self.vocabToNum = {}
        self.numToVocab = {}
        self.full_text = ''
        self.chars = []
        self.words = []
        self.vocabSize = 0
        self.batches = []
        self.labels = []

    def loadText(self):
        with open(self.data_path) as f:
            self.full_text = f.read()

    def fileToWordList(self):
        self.loadText()
        modified_text = self.full_text.translate(str.maketrans(punctuation_translations))
        self.words = set(modified_text.split())
        return self.words

    def fileToCharList(self):
        self.loadText()
        self.chars = set(list(self.full_text))
        return self.chars

    def mapWords(self):
        self.fileToWordList()
        if self.words is not []:
            self.vocabToNum = {c:int(i) for i,c in enumerate(self.words)}
            self.numToVocab = {int(self.vocabToNum[vocab]):vocab for vocab in self.vocabToNum}
            self.vocabSize = len(self.vocabToNum)

    def mapChars(self):
        self.fileToCharList()
        if self.chars is not []:
            self.vocabToNum = {c:int(i) for i,c in enumerate(self.chars)}
            self.numToVocab = {int(self.vocabToNum[vocab]):vocab for vocab in self.vocabToNum}
            self.vocabSize = len(self.vocabToNum)
    
    def saveVocabMap(self, map_path):
        with open(map_path,'w') as output_file:
            json.dump(self.vocabToNum, output_file)

    def loadVocabMap(self, map_path):
        with open(map_path) as input_file:
            self.vocabToNum = json.load(input_file)
            self.numToVocab = {int(self.vocabToNum[vocab]):vocab for vocab in self.vocabToNum}
            self.vocabSize = len(self.vocabToNum)

    def convertVocabToNumeric(self, input_text=None):
        if input_text == None:
            input_text = self.full_text
        return  [self.vocabToNum[character] for character in input_text]

    def convertNumericToVocab(self, input_numbers):
        return [self.numToVocab[num] for num in input_numbers]

    def oneHots(self, numerical_list):
        result = nd.zeros((len(numerical_list),self.vocabSize), ctx)
        for i, num in enumerate(numerical_list):
            result[i,num] = 1.0
        return result

    def makeBatches(self, seqLength):
        batches = []
        labels = []
        numeric_list=self.convertVocabToNumeric()
        batchCount = len(numeric_list)//seqLength
        for i in range(batchCount):
            sequence = numeric_list[i*seqLength:((i+1)*seqLength)-1]
            label_text = numeric_list[i*seqLength+1:((i+1)*seqLength)]
            batches.append(self.oneHots(sequence))
            labels.append(self.oneHots(label_text))
        self.batches = batches
        self.labels = labels