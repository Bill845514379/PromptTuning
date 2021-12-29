import random
import numpy as np

class RandomSet:
    def __init__(self, probability):
        data_size = 100000
        data = []
        for i in range(data_size):
            if i < probability * data_size:
                data.append(True)
            else:
                data.append(False)
        random.shuffle(data)
        self.data = data

    def get_item(self):
        answer = random.sample(self.data, 1)
        return answer[0]


def delete_character(sentence, probability):
    random_set = RandomSet(probability)
    ans = ''
    for char in sentence:
        if char == ' ':
            ans += ' '
            continue

        select = random_set.get_item()
        if select:
            pass
        else:
            ans += char
    return ans

def delete_word(sentence, probability):
    random_set = RandomSet(probability)
    ans = ''
    for word in sentence.split(' '):
        select = random_set.get_item()
        if select:
            pass
        else:
            ans += word + ' '
    return ans

def reorder_words(sentence, probability):
    random_set = RandomSet(probability)
    ans = sentence.split(' ')

    i = 0
    while i < len(sentence.split(' '))-1:
        select = random_set.get_item()
        if select:
            tmp = ans[i]
            ans[i] = ans[i+1]
            ans[i+1] = tmp
        i += 1
    return ' '.join(ans)

def reorder_span(sentence, probability, span_radio):
    import math
    span_len = math.ceil(len(sentence) * span_radio)
    if span_len < 1:
        span_len = 1

    random_set = RandomSet(probability)

    ans = []
    for i in sentence:
        ans.append(i)

    i = 0
    while i < len(ans) - 2 * span_len:
        select = random_set.get_item()
        if select:
            tmp = ans[i:i+span_len]
            ans[i:i+span_len] = ans[i+span_len:i + 2 * span_len]
            ans[i+span_len:i + 2 * span_len] = tmp
        i += span_len
    return ''.join(ans)

if __name__ == '__main__':
    text = 'Good morning! What are you doing now? What should I do? I want to see a movie with you.'
    ans = delete_character(text, 0.1)
    print(ans)
    ans = delete_word(text, 0.1)
    print(ans)
    ans = reorder_words(text, 0.1)
    print(ans)

    ans = reorder_span(text, 0.1, 0.05)
    print(ans)

