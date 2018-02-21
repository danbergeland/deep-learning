#!/usr/bin/python3

import unittest
import data_helper as dh

data_path = "data/sample_text.txt"
charMapPath = "data/testCharMap.json"

class TestDataMethods(unittest.TestCase):
    def setUp(self):
        self.data_helper = dh.DataHelper(data_path)

    def test_testsRun(self):
        self.assertTrue(True)
    
    def test_setDataPathWithConstructor(self):
        self.assertEqual(self.data_helper.data_path,data_path)

    def test_getListOfWords(self):
        wordList = self.data_helper.fileToWordList()
        self.assertEqual(len(wordList), 83)

    def test_getListOfChar(self):
        charList = self.data_helper.fileToCharList()
        self.assertEqual(len(charList), 50)

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

if __name__ == '__main__':
    unittest.main()
 