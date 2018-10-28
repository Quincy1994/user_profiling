# coding=utf-8

import word2vec

class Myword2vec:

    def __init__(self):

        self.w2v_size = 200
        self.window = 7
        self.output_path = ""
        self.vocab_path = ""

    def train_word2vec(self, corpus_path):
        """

        :param corpus_path:
        content in corpus_path is formed as:
            "Tomorrow is better , i think i can do it . "
        :return:
        """
        print("=============== training w2v ===============")
        word2vec.word2vec(train=corpus_path, output=self.output_path, size=self.w2v_size, threads=40, cbow=0,save_vocab=self.vocab_path, verbose=True)
        print("=============== w2v has been trained ==================")


