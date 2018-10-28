# coding=utf-8
from gensim import corpora, models
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
import numpy as np


class MYLDA():
    def __init__(self):
        self.lda_dict_path = ""
        self.lda_model_path = ""
        self.lda_dim = 300

    def train_lda_amodel(self, corpus):
        """
        :param corpus:
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
        """
        print("====================== train lda model =======================")
        texts = [document.split(" ") for document in corpus]
        dictonary = corpora.Dictionary(texts)
        dictonary.save(self.lda_dict_path)
        bow_corpus = [dictonary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lda = models.LdaModel(corpus_tfidf, id2word=dictonary, num_topics=self.lda_dim)
        lda.save(self.lda_model_path)
        print("======================= lda  model has been trained ============================")

    def get_lda_vectors(self, corpus):
        """

        :param corpus:
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
             lda vectors ------ shape(nrows, lda_dim)
        """
        dictionary = corpora.Dictionary().load(self.lda_dict_path)
        lda = models.LdaModel(id2word=dictionary).load(self.lda_model_path)
        texts = [document.split(" ") for document in corpus]
        lda_vectors = np.zeros([len(texts), self.lda_dim], dtype=np.float32)
        for i, text in enumerate(texts):
            bow = dictionary.doc2bow(text)
            lda_value = lda[bow]
            for value in lda_value:
                d, v = value[0], value[1]
                lda_vectors[i][d] = v
        return lda_vectors