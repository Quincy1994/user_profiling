#coding=utf-8
import configparser
import traceback
import json
import os


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    """

    """General"""
    revision = 'flatten'
    datapath = './data/smallset/'
    embed_path = './data/embedding.txt'

    optimizer = 'adam'
    attn_mode = 'attn'
    seq_encoder = 'bigru'
    task = 'gender'

    # ====== data path =======
    cache_dir = '/media/iiip/Elements/sougou/sogou/cache'
    origin_data_path = '/media/iiip/Elements/sougou/sogou/first_train_jieba_cut.csv'

    vocab_path = cache_dir + '/vocab.pk'
    age_path = cache_dir + '/age/all_age_data.csv'
    age_train_path = cache_dir + '/age/train_age_data.csv'
    age_val_path = cache_dir + '/age/val_age_data.csv'
    age_test_path = cache_dir + '/age/test_age_data.csv'
    age_samples_logger = cache_dir + '/age/age_samples.txt'
    age_path_list = [age_path, age_train_path, age_val_path, age_test_path, age_samples_logger]

    gender_path = cache_dir + '/gender/all_gender_data.csv'
    gender_train_path = cache_dir + '/gender/train_gender_data.csv'
    gender_val_path = cache_dir + '/gender/val_gender_data.csv'
    gender_test_path = cache_dir + '/gender/test_gender_data.csv'
    gender_samples_logger = cache_dir + '/gender/gender_samples.txt'
    gender_path_list = [gender_path, gender_train_path, gender_val_path, gender_test_path, gender_samples_logger]

    edu_path = cache_dir + '/edu/all_edu_data.csv'
    edu_train_path = cache_dir + '/edu/train_edu_data.csv'
    edu_val_path = cache_dir + '/edu/val_edu_data.csv'
    edu_test_path = cache_dir + '/edu/test_edu_data.csv'
    edu_samples_logger = cache_dir + '/edu/edu_samples.txt'
    edu_path_list = [edu_path, edu_train_path, edu_val_path, edu_test_path, edu_samples_logger]

    # ======= input parameter =========
    word_seq_maxlen = 800
    num_class = 2


    # ====== training parameter =======
    max_epochs = 20
    max_first_epochs = 10
    max_second_epochs = 20
    batch_sz = 64
    embed_size = 128
    lr = 0.001
    decay_steps = 10000
    decay_rate = 0.9
    istrain = True
    dropout_keep_prob = 0.5
    pre_trained = False
    early_stopping = 5
    net_state = 3
    reg = 0.

    # ====== network layer ==========

    ## cnn_layer
    num_filters = 64
    filers_sizes = [3,4,5]

    ## capsule layer
    out_caps_num = 10
    out_caps_dim = 100
    rout_iter = 3

    ## rnn_layer
    hidden_size = 50
    grad_clip = 5

    ## atten layer
    atten_size = hidden_size * 2

    ## gate_layer
    gate_size = hidden_size * 2

    dropout = 0.2


    def __init__(self):
        self.attr_list = [i for i in list(Config.__dict__.keys()) if
                          not callable(getattr(self, i)) and not i.startswith("__")]

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
                raise ValueError('something wrong in “%s” entry' % attr)

        with open(filePath, 'w') as fd:
            cfg.write(fd)

    def loadConfig(self, filePath):

        cfg = configparser.ConfigParser()
        cfg.read(filePath)
        gen_sec = cfg['General']
        for attr in self.attr_list:
            try:
                val = json.loads(gen_sec[attr])
                assert type(val) == type(getattr(self, attr)), \
                    'type not match, expect %s got %s' % \
                    (type(getattr(self, attr)), type(val))

                setattr(self, attr, val)
            except Exception as e:
                traceback.print_exc()
                raise ValueError('something wrong in “%s” entry' % attr)

        with open(filePath, 'w') as fd:
            cfg.write(fd)

    def set_class(self):
        if self.task == 'age':
            self.num_class = 6
        elif self.task == 'gender':
            self.num_class = 2
        elif self.task == 'education':
            self.num_class = 6
        else:
            raise ValueError("type of task is invalid")

config = Config()