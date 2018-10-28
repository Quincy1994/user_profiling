#coding=utf-8

import os
import numpy as np

class MyDoc2vecC:

    def __init__(self):
        self.cbow = 1
        self.size = 300
        self.window = 5
        self.negative = 5
        self.hs = 0
        self.sample = 0
        self.threads = 40
        self.binary = 0
        self.iter = 20
        self.min_count = 5
        self.sentence_sample = 0.1
        self.vocab_path = ""
        self.word_path = ""
        self.output_path = ""

    def train_doc2vecC(self, corpus_path, test_path):
        """

        :param corpus_path:
        content of corpus_path is formed as:

            "I am good\nYou are the best\nHow is going\n"

        :return:
        """
        # notice that "corpus_path" and "test_path" can refer to the same file, but
        # vectors in output_path are accordance with the test_path

        """parameter"""
        train = " -train " + corpus_path
        word = " -word " + self.word_path
        output_path = " -output " + self.output_path
        cbow = " -cbow " + str(self.cbow)
        size = " -size " + str(self.size)
        window = " -window " + str(self.window)
        negative = " -negative " + str(self.negative)
        hs = " -hs " + str(self.hs)
        sample = " -sample " + str(self.sample)
        thread = " -thread " + str(self.threads)
        binary = " -binary " + str(self.binary)
        iter = " -iter " + str(self.iter)
        min_count = "-min-count " + str(self.min_count)
        test = " -test " + str(test_path)
        sentence_sample = " -sentence-sample " + str(self.sentence_sample)
        save_vocab = " -save-vocab " + str(self.vocab_path)

        train_command = "time ./doc2vecc" + train + word + output_path + cbow + size + window \
                        + negative + hs + sample + thread + binary + iter + min_count + test + sentence_sample + save_vocab

        os.system("gcc doc2vec.c -o doc2vecc -lm -pthread -03 -march=native -funroll -loops")
        os.system(train_command)
        os.system("rm doc2vecc")

    def get_doc2vecC_vectors(self):
        rows = open(self.output_path).readlines()
        nums = len(rows) - 1
        vectors = np.zeros([nums, self.size], dtype=np.float32)
        for i in range(nums):
            values = rows[i].strip().split(" ")
            for j in range(self.size):
                vectors[i][j] = float(values[j])
        return vectors
