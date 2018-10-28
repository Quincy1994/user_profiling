# coding=utf-8

import configparser
import traceback
import json
import os
import pickle

class Config(object):

    """
    Ths config class is used to store various hyperparameteres and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    task = 'gender'
    revision = 'None'
    model_name = "HAN"
    num_class = 2

    # ============ data path ==============
    cache_dir = "/media/iiip/Elements/user_profiling/data/follow"
    vocab_path = cache_dir + "/vocab.pk"
    train_data_path = cache_dir + "/" + task + "/train_follow.csv"
    val_data_path = cache_dir + "/" + task + "/valid_follow.csv"
    test_data_path = cache_dir + "/" + task + "/test_follow.csv"

    word_vocab = pickle.load(open(config.vocab_path, 'rb'))

    # ============ input parameter =============
    maxsent = 100 # ----------------
    maxword = 30 # ----------------
    maxseq = 2000
    ute_sent_encode = 'bigru'
    ute_topic_encode = 'bilstm'
    han_sent_encode = 'bigru'
    han_doc_encode ='bigru'
    ute_doc_encode = 'bigru'
    lstm_sent_encode = 'bilstm'
    dropout = 0.5
    isdroput = True

    # ============ topic parameter ==================
    topic_data_path = cache_dir + "/100_topic_top_100.csv"
    topic_num = 100
    topic_words = 50
    topic_encode = 'bigru'
    is_trans = True

    # ============ pretrained embedding ==============
    fix_emb = False
    W_emb_path = cache_dir + "/W_emb.pk"
    emb_size = 200
    relu_w = False

    # ============= cnn parameter ======================
    filter_sizes = [3,4,5]
    num_filters = 64

    # =============== rnn parameter =================
    word_hidden = 50
    sent_hidden = 100
    atten_size = 100

    # ============== dense parameter ================
    dense_hidden = [64, num_class]

    # ============= training parameter ===============
    lr = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    optimizer = 'adam'
    max_epochs = 20
    batch_sz = 64
    early_stopping = 5
    reg = 0.
    grad_clip = 5

    def __init__(self):
        self.attr_list = [i for i in list(Config.__dict__.keys()) if
                          not callable(getattr(self, i)) and not i.startswith("___")]

    def printall(self):
        for attr in self.attr_list:
            print(attr, getattr(self, attr), type(getattr(self, attr)))

    def saveConfig(self, filePath):

        cfg = configparser.ConfigParser()
        cfg['General'] = {}
        gen_sec = cfg['General']
        for attr in self.attr_list:
            try:
                gen_sec[attr] = json.dumps(getattr(self, attr))
            except Exception as e:
                traceback.print_exc()
                raise ValueError('something wrong in "%s" entry' % attr)

        with open(filePath, 'w') as fd:
            cfg.write(fd)

    def loadConfig(self, filePath):

        cfg = configparser.ConfigParser()
        cfg.read(filePath)
        gen_sec = cfg['General']
        for attr in self.attr_list:
            try:
                val = json.loads(gen_sec[attr])
                assert type(val) == type(getattr(self, attr)),\
                        'type not match, expect %s got %s' %\
                            (type(getattr(self, attr)), type(val))
                setattr(self, attr, val)
            except Exception as e:
                # traceback.print_exc()
                print ('something wrong in "%s" entry ' % attr)
                continue

        with open(filePath, 'w') as fd:
            cfg.write(fd)

    def set_class(self):
        if self.task == 'gender':
            self.num_class = 2
        elif self.task == 'age':
            self.num_class = 4
        else:
            raise ValueError("type of task is invalid")

config = Config()
