import torch
import numpy as np

class ScorePRF:
    def __init__(self):
        self.TP, self.FP, self.FN = 0.0, 0.0, 0.0
        self.P, self.R, self.F1 = 0.0, 0.0, 0.0

    def cal_tp_fp_fn(self, gold_label, pre_label, label):

        if type(gold_label) == np.ndarray:
            position_gold = np.nonzero(gold_label == label)
            position_pre = np.nonzero(pre_label == label)
            position_pre = np.reshape(position_pre, [-1, 1])
            position_gold = np.reshape(position_gold, [-1, 1])
        else:
            position_gold = torch.nonzero(gold_label == label)
            position_pre = torch.nonzero(pre_label == label)

        for i in position_pre:
            j = i[0]
            if gold_label[j] == label:
                self.TP += 1
            else:
                self.FP += 1

        for i in position_gold:
            j = i[0]
            if pre_label[j] != label:
                self.FN += 1

    def cal_label_f1(self):
        if self.TP + self.FP != 0:
            self.P = self.TP / (self.TP + self.FP)

        if self.TP + self.FN != 0:
            self.R = self.TP / (self.TP + self.FN)

        if self.P + self.R != 0:
            self.F1 = 2 * self.P * self.R / (self.P + self.R)

        # print(self.TP, self.FP, self.FN)
        return self.P, self.R, self.F1


class ScorePRFMulClass():
    def __init__(self, num_class):
        self.num_class = num_class
        self.metric_dic = {}
        for i in range(num_class):
            self.metric_dic[i] = ScorePRF()

    def cal_tp_fp_fn(self, gold_label, pre_label):
        for i in range(self.num_class):
            self.metric_dic[i].cal_tp_fp_fn(gold_label, pre_label, i)

    def cal_label_f1(self):
        for i in range(self.num_class):
            self.metric_dic[i].cal_label_f1()

    def cal_macro_f1_1(self):
        mean_p, mean_r, mean_f = 0, 0, 0
        for i in range(self.num_class):
            i_p, i_r, i_f = self.metric_dic[i].cal_label_f1()
            mean_p += i_p
            mean_r += i_r
        mean_p /= self.num_class
        mean_r /= self.num_class
        f1_macro = 0
        if mean_p + mean_r != 0:
            f1_macro = 2 * mean_p * mean_r / (mean_p + mean_r)
        return mean_p, mean_r, f1_macro

    def cal_macro_f1_2(self):
        mean_p, mean_r, mean_f = 0, 0, 0
        for i in range(self.num_class):
            i_p, i_r, i_f = self.metric_dic[i].cal_label_f1()
            mean_p += i_p
            mean_r += i_r
            try:
                mean_f += 2 * i_p * i_r / (i_p + i_r)
            except:
                pass
        mean_p /= self.num_class
        mean_r /= self.num_class
        mean_f /= self.num_class
        return mean_p, mean_r, mean_f


if __name__ == '__main__':
    import torch
    # pre_bio = torch.tensor([1, 0, 0, 1, 1, 1, 0, 1, 2, 3, 3, 2, 3])
    # gold_bio = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 2, 2, 2, 2, 3])
    pre_bio = np.array([1,   1,  1, 2, 0, 2, 3, 3, 1])
    gold_bio = np.array([1,   1,  0, 2, 0, 1, 3 , 1, 3])
    # score = ScorePRFMulClass(3)
    # score.cal_tp_fp_fn(gold_bio, pre_bio)
    # p, r, f = score.cal_macro_f1_1()
    # print(p, r, f)
    # p, r, f = score.cal_macro_f1_2()
