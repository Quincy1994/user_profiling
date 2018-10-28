# coding=utf-8
import gensim
from gensim.models import Doc2Vec
import numpy as np

class MYDM():

    def __init__(self):
        self.dm_model_path = ""
        self.dm_dim = 300
        self.dm = 1
        self.min_count = 5
        self.window = 5
        self.sample = 1e-3
        self.negative = 5
        self.workers = 4
        self.epochs = 20

    def train_dm(self, corpus):
        TaggedDocument = gensim.models.doc2vec.TaggedDocument
        train_doc = []
        for i, doc in enumerate(corpus):
            train_doc.append(TaggedDocument(doc, tags=[i]))
        model_dm = gensim.models.Doc2Vec(train_doc, min_count=self.min_count, size=self.dm_dim, sample=self.sample, negative=self.negative, workers=self.workers, dm=self.dm)
        print("=============== training DM model ===================")
        model_dm.train(train_doc, total_examples=model_dm.corpus_count, epochs=self.epochs)
        model_dm.save(self.dm_model_path)
        print("=================== DM model has been trained ==================")

    def get_dm_vectors(self, corpus):
        model = Doc2Vec.load(self.dm_model_path)
        texts = [document.split(" ") for document in corpus]
        dm_vectors = []
        for text in texts:
            v = model.infer_vector(text)
            dm_vectors.append(v)
        dm_vectors = np.array(dm_vectors)
        return dm_vectors

