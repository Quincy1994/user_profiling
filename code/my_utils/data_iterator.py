# coding=utf-8
import numpy as np
from Config import config
import pickle
from keras.preprocessing import sequence
import pandas as pd


word_vocab = pickle.load(open(config.vocab_path, 'rb'))


def to_categorical(labels):

    y = []
    for label in labels:
        y_line = np.zeros(shape=(config.num_class), dtype=np.int32)
        y_line[label -1] = 1
        y.append(y_line)
    y = np.array(y)
    return y

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1) * batch_size)) for i in range(0, nb_batch)]


def batch_generator(contents, labels, batch_size=128, shuffle=True):

    sample_size = contents.shape[0]
    index_array = np.arange(sample_size)

    if shuffle:
        np.random.shuffle(index_array)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start: batch_end]
        batch_contents = contents[batch_ids]
        batch_x, batch_wNum = preprocess(batch_contents)
        batch_labels = to_categorical(labels[batch_ids])
        yield (batch_x, batch_wNum, batch_labels)






def preprocess(contents, word_maxlen=config.word_seq_maxlen):
    unknow_index = word_vocab['UNK']
    word_r = []
    wNum = []
    for content in contents:
        word_c = []
        content = content.lower().strip()
        words = content.split(",")
        wd_num = len(words)
        wd_num = min(wd_num, word_maxlen)
        wNum.append(wd_num)
        for word in words:
            if word in word_vocab:
                index = word_vocab[word]
            else:
                index = unknow_index
            word_c.append(index)
        word_c = np.array(word_c)
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=word_maxlen, padding="post", truncating="post", value=0)
    wNum = np.array(wNum)
    return word_seq, wNum

def get_train_data(shuffle=True):
    if config.task == 'age':
        age_train = pd.read_csv(config.age_train_path, sep='\t')
        contents = age_train['content']
        labels = age_train['age']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    elif config.task == 'gender':
        gender_train = pd.read_csv(config.gender_train_path, sep='\t')
        contents = gender_train['content']
        # print(print(contents.shape[0]))
        labels = gender_train['gender']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    elif config.task == 'education':
        edu_train = pd.read_csv(config.edu_train_path, sep='\t')
        contents = edu_train['content']
        labels = edu_train['education']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    else:
        raise ValueError('task %s does not exist' % (config.task))
    return batches

def get_val_data(shuffle=False):
    if config.task == 'age':
        age_data = pd.read_csv(config.age_val_path,sep='\t')
        contents = age_data['content']
        labels = age_data['age']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    elif config.task == 'gender':
        gender_data = pd.read_csv(config.gender_val_path, sep='\t')
        contents = gender_data['content']
        labels = gender_data['gender']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    elif config.task == 'education':
        edu_data = pd.read_csv(config.edu_val_path, sep='\t')
        contents = edu_data['content']
        labels = edu_data['education']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    else:
        raise ValueError('task %s does not exist' % (config.task))
    return batches

def get_test_data(shuffle=False):
    if config.task == 'age':
        age_data = pd.read_csv(config.age_test_path,sep='\t')
        contents = age_data['content']
        labels = age_data['age']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    elif config.task == 'gender':
        gender_data = pd.read_csv(config.gender_test_path, sep='\t')
        contents = gender_data['content']
        labels = gender_data['gender']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    elif config.task == 'education':
        edu_data = pd.read_csv(config.edu_test_path, sep='\t')
        contents = edu_data['content']
        labels = edu_data['education']
        batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    else:
        raise ValueError('task %s does not exist' % (config.task))
    return batches


# batches = get_test_data()
# for batch in batches:
#     content = batch[0]
#     print (content)















