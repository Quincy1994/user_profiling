# coding=utf-8
import numpy as np
from Config import config
from keras.preprocessing import sequence
import pandas as pd



def han_paddata(data_x, max_snt_num, max_wd_num):

    """
    :param data_x: content
    :param max_snt_num:
    :param max_wd_num:
    :return:
    """
    snt_num = np.array([min(len(doc.split("|||")), max_snt_num) for doc in data_x], dtype=np.int32)
    snt_sz = min(np.max(snt_num), max_snt_num) # truncate those lengths larger than max_snt_num

    wd_num =  [[len(sent.split(" ")) for sent in doc.split("|||")] for doc in data_x]
    # wd_sz = min(max(map(max, wd_num)), max_wd_num) # truncate those lengths larger than max_wd_num

    sNum = snt_num
    wNum = np.zeros(shape=[len(data_x), max_snt_num], dtype=np.int32)

    for i, document in enumerate(data_x):
        for j, sentence in enumerate(document.split("|||")):
            if j >= snt_sz:
                continue
            wNum[i,j] = min(wd_num[i][j], max_wd_num)
    # print(wNum)
    return sNum, wNum

def cnn_paddata(data_x, max_seq):
    snt_num = np.array([min(len(doc.replace("|||", " ").split(" ")), max_seq) for doc in data_x], dtype=np.int32)

    return snt_num


def to_categorical(labels):

    y = []
    for label in labels:
        y_line = np.zeros(shape=(config.num_class), dtype=np.int32)
        y_line[label] = 1
        y.append(y_line)
    y = np.array(y)
    return y

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size))) # 向上取整
    return [(i*batch_size, min(size, (i+1) * batch_size)) for i in range(0, nb_batch)]

def get_word_seq(contents, word_maxlen=config.maxword, mode="post"):
    unknow_index = len(config.word_vocab) - 1
    word_r = []
    for content in contents:
        word_c = []
        content = content.lower().strip()
        words = content.split(" ")
        for word in words:
            if word in config.word_vocab:
                index = config.word_vocab[word]
            else:
                index = unknow_index
            word_c.append(index)
        word_c = np.array(word_c)
        word_r.append(word_c)
    word_seq = sequence.pad_sequences(word_r, maxlen=word_maxlen, padding=mode, truncating=mode, value=0)
    return word_seq

def get_han_repreprocess(contents, sentence_num=config.maxsent, sentence_length=config.maxword):
    contents_seq = np.zeros(shape=(len(contents), sentence_num, sentence_length))
    for index, content in enumerate(contents):
        if index >= len(contents): break
        sentences = content.split("|||")
        word_seq = get_word_seq(sentences, word_maxlen=config.maxword)
        word_seq = word_seq[:sentence_num]
        contents_seq[index][:len(word_seq)] = word_seq
    return contents_seq

def get_cnn_repreprocess(contents, sentence_seq=config.maxseq):
    merge_contents = []
    for index, content in enumerate(contents):
        content = content.replace("|||", " ")
        merge_contents.append(content)
    contents_seq = get_word_seq(merge_contents, word_maxlen=sentence_seq)
    return contents_seq


def get_topic_repreprocess(topic_contents, topic_words_num=config.topic_words):
    topic_seq = get_word_seq(topic_contents, word_maxlen=topic_words_num) # shape: (topic_num, topic_words_num)
    return topic_seq

def get_topic_data():
    data = pd.read_csv(config.topic_data_path, sep='\t')
    topic_contents = data["word"]
    topic_seq = get_topic_repreprocess(topic_contents, config.topic_words)
    return topic_seq


def han_batch_generator(contents, labels, batch_size=64, shuffle=True):

    sample_size = contents.shape[0]
    print(sample_size)
    index_array = np.arange(sample_size) # 返回一个有终点和起点的固定步长的排列

    if shuffle:
        np.random.shuffle(index_array)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start: batch_end]
        batch_contents = contents[batch_ids]
        batch_sNum, batch_wNum = han_paddata(batch_contents, config.maxsent, config.maxword)
        batch_labels = to_categorical(labels[batch_ids])
        batch_x = get_han_repreprocess(contents=batch_contents, sentence_num=config.maxsent, sentence_length=config.maxword)
        yield (batch_x, batch_sNum, batch_wNum, batch_labels)

def cnn_batch_generator(contents, labels, batch_size=64, shuffle=True):
    sample_size = contents.shape[0]
    index_array = np.arange(sample_size)
    if shuffle:
        np.random.shuffle(index_array)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start: batch_end]
        batch_contents = contents[batch_ids]
        batch_x = get_cnn_repreprocess(contents=batch_contents, sentence_seq=config.maxseq)
        batch_labels = to_categorical(labels[batch_ids])
        batch_sNum = cnn_paddata(batch_contents, config.maxseq)
        yield (batch_x, batch_sNum, batch_labels)

def get_han_data(data_path, shuffle=True, batch_generator=han_batch_generator):
    data = pd.read_csv(data_path, sep='\t')
    contents = data['content']
    labels = data['label']
    batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    return batches

def get_cnn_data(data_path, shuffle=True, batch_generator=cnn_batch_generator):
    data = pd.read_csv(data_path, sep='\t')
    contents = data['content']
    labels = data['label']
    batches = batch_generator(contents, labels, batch_size=config.batch_sz, shuffle=shuffle)
    return batches