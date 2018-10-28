# coding=utf-8
import pandas as pd
from collections import defaultdict
import pickle
import word2vec
import numpy as np

class Myvocab:

    def __init__(self):

        # vocab
        self.vocab_path = ""
        self.min_count = 5
        self.vocab_path = ""

        # word embedding
        self.w2v_size = 200
        self.window = 7
        self.w2v_emb_path = ""
        self.w2v_mat_path = ""

    def create_vocab(self, data_path, save_vocab_path):

        """
        data_path is a csv file
        content in data_path is formed as:
        | user_id | label | content |

        :param save_vocab_path:
        :return:
        """

        data = pd.read_csv(data_path, sep='\t')
        contents = data['content']
        nums = len(contents)

        word_freq = defaultdict(int)
        for i in range(nums):
            content = contents[i].strip().split(" ")
            for word in content:
                word_freq[word] += 1

        w_vocab = {}
        index = 1
        for w in word_freq:
            freq = word_freq[w]
            if freq >= self.min_count:
                w_vocab[w] = index
                index += 1
        w_vocab['UNK'] = index
        print("len of vocab size", len(w_vocab))

        with open(save_vocab_path, 'wb') as f:
            pickle.dump(w_vocab, f)

    def train_word2vec(self, corpus_path):
        """

        :param corpus_path:
        content in corpus_path is formed as:
            "Tomorrow is better , i think i can do it . "
        :return:
        """
        print("=============== training w2v ===============")
        word2vec.word2vec(train=corpus_path, output=self.w2v_emb_path, size=self.w2v_size, threads=40, cbow=0, save_vocab=self.vocab_path, verbose=True)
        print("=============== w2v has been trained ==================")

    def create_emb_matrix(self):

        vocab = pickle.load(open(self.vocab_path, "rb"))
        model = word2vec.load(self.w2v_emb_path)
        emb_size = len(model['UNK'])
        word_emb = [np.random.uniform(0, 0, emb_size) for j in range(len(vocab) + 1)]

        num = 0
        for word in vocab:
            index = vocab[word]
            if word in model:
                word_emb[index] = np.array(model[word])
                num += 1
            else:
                word_emb[index] = np.random.uniform(-0.01, 0.01, emb_size)
        word_emb = np.array(word_emb)
        print("valid word number:", num)
        print("vocab size: ", len(vocab))

        with open(self.w2v_mat_path, 'wb') as f:
            pickle.dump(word_emb, f)
            print("word embedding matrix has been finished")



































