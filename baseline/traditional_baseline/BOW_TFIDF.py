# coding=utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np

class MYTFIDF():

    def __init__(self):
        self.max_feature = 50000         # max features of tfidf according to term-frequency
        self.min_df = 1
        self.ngram_range = (1, 1)
        # self.tfidf_model_path = "./tfidf_model.m"  # take care of the store path


    def train_tfidf(self, corpus):

        """
        :param corpus: list
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
        """
        print("================== training tfidf model ==========================")
        tfidf = TfidfVectorizer(max_features=self.max_feature, min_df=self.min_df, ngram_range=self.ngram_range).fit(corpus)
        joblib.dump(tfidf, self.tfidf_model_path)
        print("=============== tfidf model has been trained =======================")

    def get_tfidf_vector(self, corpus):
        """
        :param corpus: list
            form of corpus:
                [   "I am good",
                    "How are you",
                    "That's OK"
                ]
        :return:
            tfidf vectors ---- shape(nrows, max_features)
        """
        tfidf_model = joblib.load(self.tfidf_model_path)
        tfidf_vectors = tfidf_model.transform(corpus).toarray()
        return tfidf_vectors


# usage example
def useage_example():
    corpus = ["I am good", "How are you", "That is ok"]
    tfidf = MYTFIDF()
    tfidf.train_tfidf(corpus)
    tfidf_vectors = tfidf.get_tfidf_vector(corpus)
    print(np.shape(tfidf_vectors))
