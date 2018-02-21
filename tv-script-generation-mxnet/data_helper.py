#!/usr/bin/python3

import string
import json

punctuation_translations = {
    '.' : ' PERIOD ',
    ',' : ' COMMA ',
    '!' : ' EXCLAMATION ',
    '(' : ' LEFTPAREN ',
    ')' : ' RIGHTPAREN ',
    '?' : ' QUESTIONMARK ',
}

class DataHelper():
    def __init__(self, data_path):
        self.data_path = data_path
        self.vocabToNum = {}
        self.numToVocab = {}
        self.chars = []
        self.words = []

    def fileToWordList(self):
        with open(self.data_path) as f:
            full_text = f.read()
            full_text = full_text.translate(str.maketrans(punctuation_translations))
            self.words = set(full_text.split())
            return self.words

    def fileToCharList(self):
        with open(self.data_path) as f:
            full_text = f.read()
            self.chars = set(list(full_text))
            return self.chars

    def mapWords(self):
        self.fileToWordList()
        if self.words is not []:
            self.vocabToNum = {c:int(i) for i,c in enumerate(self.words)}
            self.numToVocab = {int(self.vocabToNum[vocab]):vocab for vocab in self.vocabToNum}

    def mapChars(self):
        self.fileToCharList()
        if self.chars is not []:
            self.vocabToNum = {c:int(i) for i,c in enumerate(self.chars)}
            self.numToVocab = {int(self.vocabToNum[vocab]):vocab for vocab in self.vocabToNum}
    
    def saveVocabMap(self, map_path):
        with open(map_path,'w') as output_file:
            json.dump(self.vocabToNum, output_file)

    def loadVocabMap(self, map_path):
        with open(map_path) as input_file:
            self.vocabToNum = json.load(input_file)
            self.numToVocab = {int(self.vocabToNum[vocab]):vocab for vocab in self.vocabToNum}
