# -*- coding: UTF-8 -*-
import pandas as pd
from common.util import change_lr
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from common.text2id import X_data2id, get_answer_id
import os
import torch
from config.cfg import cfg, path, hyper_roberta
from common.load_data import load_data, tokenizer, data_split
from model.PromptMask import PromptMask, LMHead
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.autograd import Variable
from common.metric import ScorePRF
from common.set_random_seed import setup_seed
import time

seeds = [10, 100, 1000, 2000, 4000]
average_acc = 0
for test_id in range(len(seeds)):
    print('~~~~~~~~~~~~~ 第', test_id+1,'次测试 ~~~~~~~~~~~~~~~~~~~')
    setup_seed(seeds[test_id])
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['gpu_id'])
    device = torch.device(cfg['device'])

    pos_X, pos_y = load_data(path['pos_path'])
    train_pos_X, train_pos_y, test_pos_X, test_pos_y = data_split(pos_X, pos_y, cfg['K'], cfg['Kt'])
    train_pos_X, test_pos_X = X_data2id(train_pos_X, tokenizer), X_data2id(test_pos_X, tokenizer)
    train_pos_y, test_pos_y = get_answer_id(train_pos_y, tokenizer), get_answer_id(test_pos_y, tokenizer)

    neg_X, neg_y = load_data(path['neg_path'])
    train_neg_X, train_neg_y, test_neg_X, test_neg_y = data_split(neg_X, neg_y, cfg['K'], cfg['Kt'])
    train_neg_X, test_neg_X = X_data2id(train_neg_X, tokenizer), X_data2id(test_neg_X, tokenizer)
    train_neg_y, test_neg_y = get_answer_id(train_neg_y, tokenizer), get_answer_id(test_neg_y, tokenizer)

    train_X = torch.tensor(np.vstack([train_pos_X, train_neg_X]))
    train_y = torch.tensor(np.hstack([train_pos_y, train_neg_y]))
    test_X = torch.tensor(np.vstack([test_pos_X, test_neg_X]))
    test_y = torch.tensor(np.hstack([test_pos_y, test_neg_y]))

    train_data = TensorDataset(train_X, train_y)
    test_data = TensorDataset(test_X, test_y)

    loader_train = DataLoader(
        dataset=train_data,
        batch_size=cfg['train_batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    loader_test = DataLoader(
        dataset=test_data,
        batch_size=cfg['test_batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    net_mask = PromptMask()
    net_mask = net_mask.to(device)

    net_head = LMHead()
    net_head = net_head.to(device)
    # 核心思想： 将BERT部分参数与其他参数分开管理
    # bert_params = []
    # other_params = []
    #
    # for name, para in net.named_parameters():
    #     # 对于需要更新的参数：
    #     if para.requires_grad:
    #         # BERT部分所有参数都存在于bert_encoder中（针对不同模型，可以print(name)输出查看）
    #         # print(name)
    #         if "roberta.encoder" in name:
    #             bert_params += [para]
    #         else:
    #             other_params += [para]
    #
    # params = [
    #     {"params": bert_params, "lr": cfg['bert_learning_rate']},
    #     {"params": other_params, "lr": cfg['other_learning_rate']},
    # ]

    if cfg['optimizer'] == 'Adam':
        optimizer_mask = optim.Adam(net_mask.parameters(), lr=cfg['bert_learning_rate'])
        optimizer_head = optim.Adam(net_mask.parameters(), lr=cfg['other_learning_rate'])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True,
    #                                                        threshold=0.0001,
    #                                                        threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    epoch = cfg['epoch']
    print(cfg)
    print(hyper_roberta)

    for i in range(epoch):
        # if i > 5:
        #     current_lr *= 0.95
        #     change_lr(optimizer, current_lr)

        if (i+1) % 10 != 0:
            change_lr(optimizer_mask, 0)
        else:
            change_lr(optimizer_mask, cfg['bert_learning_rate'])

        print('-------------------------   training   ------------------------------')
        time0 = time.time()
        batch = 0
        ave_loss, sum_acc = 0, 0
        for batch_x, batch_y in loader_train:
            net_mask.train()
            net_head.train()
            batch_x, batch_y = Variable(batch_x).long(), Variable(batch_y).long()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            output_0 = net_mask(batch_x)
            output = net_head(output_0)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, batch_y)
            loss.backward()

            optimizer_mask.step()  # 更新权重
            optimizer_head.step()  # 更新权重

            optimizer_mask.zero_grad()  # 清空梯度缓存
            optimizer_head.zero_grad()  # 清空梯度缓存
            ave_loss += loss
            batch += 1

            if batch % 2 == 0:
                print('epoch:{}/{},batch:{}/{},time:{}, loss:{},bert_learning_rate:{}, other_learning_rate:{}'.format(i + 1, epoch, batch,
                                                                                         len(loader_train),
                                                                                         round(time.time() - time0, 4),
                                                                                         loss,
                                                                                         optimizer_mask.param_groups[0]['lr'],
                                                                                        optimizer_head.param_groups[0]['lr']))
        # scheduler.step(ave_loss)
        print('------------------ epoch:{} ----------------'.format(i + 1))
        print('train_average_loss{}'.format(ave_loss / len(loader_train)))
        print('============================================'.format(i + 1))

        time0 = time.time()
        if (i + 1) % epoch == 0:
            label_out, label_y = [], []
            print('-------------------------   test   ------------------------------')
            sum_acc, num = 0, 0
            # torch.save(net.state_dict(), 'save_model/params' + str(i + 1) + '.pkl')
            for batch_x, batch_y in loader_test:
                net_mask.eval()
                net_head.eval()

                batch_x, batch_y = Variable(batch_x).long(), Variable(batch_y).long()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                with torch.no_grad():
                    output = net_mask(batch_x)
                    output = net_head(output)

                _, pred = torch.max(output, dim=1)

                pred = pred.cpu().detach().numpy()
                batch_y = batch_y.cpu().detach().numpy()


                for j in range(pred.shape[0]):
                    label_out.append(pred[j])
                    label_y.append(batch_y[j])

            label_out = np.array(label_out)
            label_y = np.array(label_y)

            acc = (np.sum(label_y == label_out)) / len(label_y)
            print('------------------ epoch:{} ----------------'.format(i + 1))
            print('test_acc:{}, time:{}'.format(round(acc, 4), time.time()-time0))
            print('============================================'.format(i + 1))
            average_acc += acc


average_acc /= 5

print('average_acc:{}'.format(round(average_acc, 4),))
