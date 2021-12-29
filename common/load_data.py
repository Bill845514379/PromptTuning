# -*- coding: UTF-8 -*-

from config.cfg import path, cfg
from common.util import delete_character, delete_word, reorder_span, reorder_words
from pytorch_transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(path['roberta_path'])

def load_data(path):
    data_X, y = [], []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = line.split('|')
            data_X.append(line[0])
            y.append(int(line[1]))
    return data_X, y

def get_random_sample_ids(length, K):
    import random
    ids_list = []
    for i in range(length):
        ids_list.append(i)
    ids = random.sample(ids_list, K)
    return ids

def data_split(data_X, data_y, K=8, Kt=1000):
    train_ids = get_random_sample_ids(len(data_X), K)
    train_X, train_y = [], []
    test_all_X, test_all_y = [], []
    for i in range(len(data_X)):
        if i in train_ids:
            train_X.append('<s> ' + data_X[i] + ' </s> </s> ' + cfg['template'] + ' </s>')
            # print('<s> ' + data_X[i] + ' </s> </s> ' + cfg['template'] + ' </s>')
            train_y.append(data_y[i])
            # train_X.append('<s>' + delete_word(data_X[i], 0.15) + '</s> </s> ' + cfg['template'] + ' </s>')
            # train_y.append(data_y[i])
            # train_X.append('<s>' +reorder_words(data_X[i], 0.15) + '</s> </s> ' + cfg['template'] + ' </s>')
            # train_y.append(data_y[i])
        else:
            test_all_X.append('<s> ' + data_X[i] + ' </s> </s> ' + cfg['template'] + ' </s>')
            test_all_y.append(data_y[i])

    test_ids = get_random_sample_ids(len(test_all_X), Kt)
    test_X, test_y = [], []
    for i in range(len(test_all_X)):
        if i in test_ids:
            test_X.append(test_all_X[i])
            test_y.append(test_all_y[i])

    return train_X, train_y, test_X, test_y





