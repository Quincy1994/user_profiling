# coding=utf-8
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

from Config import config
import pickle

def get_data(data_path, sep='\t'):
    data = pd.read_csv(data_path, sep=sep)
    return data

def get_vocab():
    vocab = pickle.load(open(config.vocab_path, 'rb'))
    return vocab

def create_word_vocab(overwriter=False):
    word_freq = defaultdict(int)

    all_data = get_data(config.origin_data_path, sep=',')
    content = all_data['query_content']
    print (len(content))

    for line in content:
        line = line.lower().strip().replace(',,', ',')
        # print (line)
        words = line.split(",")
        for word in words:
            # print(word)
            if " " == word or "" == word:
                continue
            word_freq[word] += 1

    vocab = {}
    i = 1
    min_freq = 1
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = i
            i += 1
    vocab['UNK'] = i+1
    # print(vocab)
    print("size of vocab:", len(vocab))
    #
    if overwriter:
        with open(config.vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
    print("finish to create vocab")

def create_data_samples(target, save_all_samples=False, save_samples=False):
    data = []
    missing_data = []
    all_data = get_data(config.origin_data_path, sep=',')
    labels = all_data[target]
    ids = all_data['id']
    content = all_data['query_content']
    for i in range(0, len(labels), 1):
        label = labels[i]
        row = {}
        row['id'] = ids[i]
        row[target] = labels[i]
        row['content'] = content[i].lower().strip().replace(',,', ',')
        if label == 0:
            missing_data.append(row)
        else:
            data.append(row)

    print("label:", target)
    print("samples number:", len(data))
    print("missing number:", len(missing_data))

    if save_all_samples:
        data = pd.DataFrame(data)
        data = data[["id", target, "content"]]
        data.fillna("", inplace=True)
        if target == 'age':
            data.to_csv(config.age_path, index=False, sep='\t')
        elif target == 'gender':
            data.to_csv(config.gender_path, index=False, sep='\t')
        elif target == 'education':
            data.to_csv(config.edu_path, index=False, sep='\t')
        else:
            raise ValueError('wrong target label')
    if save_samples:
        train, test = train_test_split(data, test_size=0.2, shuffle=True, random_state=1)
        train, val = train_test_split(train, test_size=0.1, shuffle=True, random_state=1)
        logger = ''
        logger += "label: %s\n" %target
        logger += "samples number: %s\n"%len(data)
        logger += "missing number: %s\n"%len(missing_data)
        logger += "len of train samples is %s\n" %(len(train))
        logger += "len of val samples is %s\n" % (len(val))
        logger += "len of test samples is %s\n" % (len(test))
        print(logger)

        if target == 'age':
            data_path_list = config.age_path_list
        elif target == 'gender':
            data_path_list = config.gender_path_list
        elif target == 'education':
            data_path_list = config.edu_path_list
        else:
            raise ValueError('wrong target label')
        train.to_csv(data_path_list[1], index=False, sep='\t')
        val.to_csv(data_path_list[2], index=False, sep='\t')
        test.to_csv(data_path_list[3], index=False, sep='\t')

        sample_log = data_path_list[4]
        with open(sample_log, 'w') as log:
            log.write(logger)
            log.close()




# create_data_samples('education', save_all_samples=True, save_samples=True)








# create_word_vocab(overwriter=True)