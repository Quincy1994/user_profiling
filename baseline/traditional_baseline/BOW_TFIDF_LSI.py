# coding=utf-8
from gensim import corpora, models
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
import numpy as np

class MYLSI():

    def __init__(self):
        self.lsi_dict_path = ""
        self.lsi_model_path = ""
        self.lsi_dim = 300

    def train_lsi_model(self, corpus):
        """
        :param corpus:
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
        """
        print("====================== train lsi model =======================")
        texts = [document.split(" ") for document in corpus]
        dictonary = corpora.Dictionary(texts)
        dictonary.save(self.lsi_dict_path)
        bow_corpus = [dictonary.doc2bow(text) for text in texts]
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word=dictonary, num_topics=self.lsi_dim)
        lsi.save(self.lsi_model_path)
        print("======================= lsi model has been trained ============================")

    def get_lsi_vectors(self, corpus):
        """

        :param corpus:
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
             lsi vectors ------ shape(nrows, lsi_dim)
        """
        dictionary = corpora.Dictionary().load(self.lsi_dict_path)
        lsi = models.LsiModel(id2word=dictionary).load(self.lsi_model_path)
        texts = [document.split(" ") for document in corpus]
        lsi_vectors = np.zeros([len(texts), self.lsi_dim], dtype=np.float32)
        for i, text in enumerate(texts):
            bow = dictionary.doc2bow(text)
            lsi_value = lsi[bow]
            for value in lsi_value:
                d, v = value[0], value[1]
                lsi_vectors[i][d] = v
        return lsi_vectors