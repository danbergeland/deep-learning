#!/usr/bin/python3

import unittest
import data_helper as dh
from mxnet import nd

data_path = "data/sample_text.txt"
charMapPath = "data/testCharMap.json"
SAMPLE_UNIQUE_WORDS = 83
SAMPLE_UNIQUE_CHARS = 50
SAMPLE_LENGTH_CHARS = 695

class TestDataMethods(unittest.TestCase):
    def setUp(self):
        self.data_helper = dh.DataHelper(data_path)

    def test_testsRun(self):
        self.assertTrue(True)
    
    def test_setDataPathWithConstructor(self):
        self.assertEqual(self.data_helper.data_path,data_path)

    def test_getListOfWords(self):
        wordList = self.data_helper.fileToWordList()
        self.assertEqual(len(wordList), SAMPLE_UNIQUE_WORDS)

    def test_getListOfChar(self):
        charList = self.data_helper.fileToCharList()
        self.assertEqual(len(charList), SAMPLE_UNIQUE_CHARS)

    def test_charToInt(self):
        self.data_helper.mapChars()
        a_val = self.data_helper.vocabToNum['a']
        self.assertEqual(self.data_helper.vocabToNum['a'],a_val)
        self.assertEqual(self.data_helper.numToVocab[a_val],'a')

    def test_saveAndLoadVocabMap(self):
        self.data_helper.mapChars()
        a_val = self.data_helper.vocabToNum['a']
        self.assertEqual(self.data_helper.vocabToNum['a'],a_val)
        self.assertEqual(self.data_helper.numToVocab[a_val],'a')
        self.data_helper.saveVocabMap(charMapPath)

        self.data_helper = dh.DataHelper(data_path)
        self.data_helper.loadVocabMap(charMapPath)
        self.assertEqual(self.data_helper.vocabToNum['a'],a_val)
        self.assertEqual(self.data_helper.numToVocab[a_val],'a')

    def test_mapWords(self):
        self.data_helper.mapWords()
        the_val = self.data_helper.vocabToNum['the']
        self.assertEqual(self.data_helper.vocabToNum['the'], the_val)
        self.assertEqual(self.data_helper.numToVocab[the_val], 'the')

    def test_convertTextFileToNumbers_convertBackToChars(self):
        self.data_helper.mapChars()
        numericList = self.data_helper.convertVocabToNumeric()
        for num in numericList:
            self.assertTrue(isinstance(num,int))
        self.assertEqual(len(numericList), SAMPLE_LENGTH_CHARS)
        charList = self.data_helper.convertNumericToVocab(numericList)
        for character in charList:
            self.assertTrue(isinstance(character,str))
        self.assertEqual(len(charList), SAMPLE_LENGTH_CHARS)

    def test_oneHots(self):
        self.data_helper.mapChars()
        num_list = self.data_helper.convertVocabToNumeric('test string')
        batches = self.data_helper.oneHots(num_list)
        self.assertEqual(batches.shape, (11,SAMPLE_UNIQUE_CHARS))

    def test_makeBatches(self):
        self.data_helper.mapChars()
        sequenceLength = 64
        self.data_helper.makeBatches(sequenceLength)
        self.assertEqual(len(self.data_helper.batches),SAMPLE_LENGTH_CHARS//sequenceLength)
        self.assertEqual(len(self.data_helper.batches), len(self.data_helper.labels))


if __name__ == '__main__':
    unittest.main()
 