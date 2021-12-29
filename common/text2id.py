from config.cfg import cfg
import numpy as np

def sentence2id(tokens_a, tokenizer):
    '''
    将text转为 id
    :param sentence:
    :return:
    '''
    tt = np.ones(cfg['max_len'])
    tt0 = tokenizer.convert_tokens_to_ids(tokens_a)
    tt[0:min(len(tt0), cfg['max_len'])] = tt0[0:min(len(tt0), cfg['max_len'])]
    return tt

def X_data2id(X_data, tokenizer):
    '''
    将整个数据集转换成id
    :param X_data_text:
    :return:
    '''
    X_data_id = []
    for i in range(len(X_data)):
        X_data_tokens = tokenizer.tokenize(X_data[i])
        X_data_id.append(sentence2id(X_data_tokens, tokenizer))
    return np.array(X_data_id)

def get_answer_id(y_data, tokenizer):
    text = ' '.join(cfg['answer'])
    token = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(token)
    answer_map = [ids[0], ids[1]]
    # answer_map = [0, 1]

    res = []
    for i in range(len(y_data)):
        res.append(answer_map[y_data[i]])
    return np.array(res)








