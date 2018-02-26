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
        wordList = self.data_helper.file_to_word_list()
        self.assertEqual(len(wordList), SAMPLE_UNIQUE_WORDS)

    def test_getListOfChar(self):
        charList = self.data_helper.file_to_char_list()
        self.assertEqual(len(charList), SAMPLE_UNIQUE_CHARS)

    def test_charToInt(self):
        self.data_helper.map_chars()
        a_val = self.data_helper.vocab_to_num['a']
        self.assertEqual(self.data_helper.vocab_to_num['a'],a_val)
        self.assertEqual(self.data_helper.num_to_vocab[a_val],'a')

    def test_saveAndLoadVocabMap(self):
        self.data_helper.map_chars()
        a_val = self.data_helper.vocab_to_num['a']
        self.assertEqual(self.data_helper.vocab_to_num['a'],a_val)
        self.assertEqual(self.data_helper.num_to_vocab[a_val],'a')
        self.data_helper.save_vocab_map(charMapPath)

        self.data_helper = dh.DataHelper(data_path)
        self.data_helper.load_vocab_map(charMapPath)
        self.assertEqual(self.data_helper.vocab_to_num['a'],a_val)
        self.assertEqual(self.data_helper.num_to_vocab[a_val],'a')

    def test_mapWords(self):
        self.data_helper.map_words()
        the_val = self.data_helper.vocab_to_num['the']
        self.assertEqual(self.data_helper.vocab_to_num['the'], the_val)
        self.assertEqual(self.data_helper.num_to_vocab[the_val], 'the')

    def test_convertTextFileToNumbers_convertBackToChars(self):
        self.data_helper.map_chars()
        numericList = self.data_helper.convert_vocab_to_numeric()
        for num in numericList:
            self.assertTrue(isinstance(num,int))
        self.assertEqual(len(numericList), SAMPLE_LENGTH_CHARS)
        charList = self.data_helper.convert_numeric_to_vocab(numericList)
        for character in charList:
            self.assertTrue(isinstance(character,str))
        self.assertEqual(len(charList), SAMPLE_LENGTH_CHARS)

    def test_oneHots(self):
        self.data_helper.map_chars()
        num_list = self.data_helper.convert_vocab_to_numeric('test string')
        batches = self.data_helper.one_hots(num_list)
        self.assertEqual(batches.shape, (11,SAMPLE_UNIQUE_CHARS))

    def test_makeBatches(self):
        self.data_helper.map_chars()
        sequence_length = 64
        self.data_helper.make_batches(sequence_length)
        self.assertEqual(len(self.data_helper.batches),SAMPLE_LENGTH_CHARS//sequence_length)
        self.assertEqual(len(self.data_helper.batches), len(self.data_helper.labels))
        self.assertEqual(self.data_helper.batches[0].shape,(sequence_length,SAMPLE_UNIQUE_CHARS))

    def test_textify_labels_match_batches(self):
        self.data_helper.map_chars()
        sequence_length = 64
        self.data_helper.make_batches(sequence_length)
        batch_text = self.data_helper.textify(self.data_helper.batches[0])
        sample_text_batch0 = "Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to "
        self.assertEqual(batch_text,sample_text_batch0)
        label_text = self.data_helper.textify(self.data_helper.labels[0])
        sample_text_label0 = "oe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to d"
        self.assertEqual(label_text, sample_text_label0)
        batch_text = self.data_helper.textify(self.data_helper.batches[9])
        sample_text_batch10 = "ou should not drink to forget your problems.\nBarney_Gumble: Yeah"
        self.assertEqual(batch_text,sample_text_batch10)
        label_text = self.data_helper.textify(self.data_helper.labels[9])
        sample_text_label10 = "u should not drink to forget your problems.\nBarney_Gumble: Yeah,"
        self.assertEqual(label_text, sample_text_label10)
        


if __name__ == '__main__':
    unittest.main()
 