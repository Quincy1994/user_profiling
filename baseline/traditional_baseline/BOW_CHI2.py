# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest

class MYCHI2():

    def __init__(self):
        self.max_feature = 50000         # max features of tfidf according to term-frequency
        self.cv_model_path = ""
        self.chi2_model_path = ""


    def train_chi2(self, corpus, label_y):

        """
        :param corpus: list
        form of corpus:
            [   "I am good",
                "How are you",
                "That's OK"
            ]
        :return:
        """
        print("================== train CountVector model =======================")
        cv = CountVectorizer().fit(corpus)
        print("==================== CountVector model has been trained =================")
        X_features = cv.transform(corpus).toarray()
        chi2_model = SelectKBest(chi2, k=self.max_feature).fit(X_features, label_y)
        print("===================== chi2 model has been trained =====================")
        joblib.dump(cv, self.cv_model_path)
        joblib.dump(chi2_model, self.chi2_model_path)
        print("=============== models have been trained =======================")

    def get_chi2_vector(self, corpus):
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
        cv_model = joblib.load(self.cv_model_path)
        chi2_model = joblib.load(self.chi2_model_path)
        cv_vectors = cv_model.transform(corpus)
        chi2_vectors = chi2_model.transform(cv_vectors)
        return chi2_vectors

