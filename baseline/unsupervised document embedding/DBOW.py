# coding=utf-8
import gensim
from gensim.models import Doc2Vec
import numpy as np

class MYDBOW():

    def __init__(self):
        self.dbow_model_path = ""
        self.dbow_dim = 300
        self.dbow = 0
        self.min_count = 5
        self.window = 5
        self.sample = 1e-3
        self.negative = 5
        self.workers = 4
        self.epochs = 20

    def train_dbow(self, corpus):
        TaggedDocument = gensim.models.doc2vec.TaggedDocument
        train_doc = []
        for i, doc in enumerate(corpus):
            train_doc.append(TaggedDocument(doc, tags=[i]))
        model_dbow = gensim.models.Doc2Vec(train_doc, min_count=self.min_count, size=self.dbow_dim, sample=self.sample, negative=self.negative, workers=self.workers, dm=self.dbow)
        print("=============== training DM model ===================")
        model_dbow.train(train_doc, total_examples=model_dbow.corpus_count, epochs=self.epochs)
        model_dbow.save(self.dbow_model_path)
        print("=================== DM model has been trained ==================")

    def get_dbow_vectors(self, corpus):
        model = Doc2Vec.load(self.dbow_model_path)
        texts = [document.split(" ") for document in corpus]
        dbow_vectors = []
        for text in texts:
            v = model.infer_vector(text)
            dbow_vectors.append(v)
        dbow_vectors = np.array(dbow_vectors)
        return dbow_vectors


