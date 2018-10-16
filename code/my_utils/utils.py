# coding=utf-8
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score

def readEmbedding(fileName):
    """
    Read Embedding Function
    :param fileName: file which stores the embedding
    :return: embeddings_index: a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'rb') as f:
        for line in f:
            line_uni = line.strip()
            line_uni = line_uni.decode('utf-8')
            values = line_uni.split(" ")
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                print(values, len(values))
        return embeddings_index


def mkEmbedMatrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix
    :param embed_dic:  word-embedding dictionary
    :param vocab_dic:  word-index dictionary
    :return:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) < 1:
        raise ValueError('Input dimension less than 1')
    vocab_sz = max(vocab_dic.values()) + 1
    EMBEDDING_DIM = len(list(embed_dic.values())[0])
    embedding_matrix = np.random.rand(vocab_sz, EMBEDDING_DIM).astype(np.float32) * 0.05
    valid_mask = np.ones(vocab_sz, dtype=np.bool)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            valid_mask[i] = False
    return embedding_matrix, valid_mask

def read_status(status_path):
    if not os.path.exists(status_path):
        return 'error'
    fd = open(status_path, 'r')
    time_stamp = float(fd.read().strip())
    fd.close()
    if time_stamp < 10.:
        return 'finished'
    cur_time = time.time()
    if cur_time - time_stamp < 1000.:
        return 'running'
    else:
        return 'error'

def valid_entry(save_path):

    if not os.path.exists(save_path):
        return False
    if read_status(save_path + '/status') == 'running':
        return True
    if read_status(save_path + '/status') == 'finished':
        return True
    if read_status(save_path + '/status') == 'error':
        return False
    raise ValueError('unknown error')

def score(pred, label):
    pre_score = np.mean(precision_score(label, pred, average=None))
    rec_score = np.mean(recall_score(label, pred, average=None))
    f_score = np.mean(f1_score(label, pred, average=None))
    return pre_score, rec_score, f_score