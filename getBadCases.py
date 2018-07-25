# -*- coding: utf-8 -*-
"""
    整个数据集的错误案例
"""
import os
import jieba
import re
import pandas as pd
import pickle

import torch
from torch.autograd import Variable

from data_process import data_processing
from TextCNN import TextCNN
from BiLSTM import BiLSTM
from TextCNN_BN import TextCNN_BN, TextCNN_multi_channel
from BiLSTM_b import BiLSTM_b
from CNN_BiLSTM_Concat import CNN_BiLSTM_a
from TextRCNN import TextRCNN
from TextCNN_BN_Pretrained_embed import TextCNN_BN_with_pretrained_embed

use_cuda = torch.cuda.is_available()
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'new_data')

# 一些超参数
BATCH_SIZE = 16
EMBEDDING_DIM = 128
MAXLENGTH = 50
KERNEL_SIZES = [3, 4, 5]
KERNEL_NUM = 128 # 卷积核数量
HIDDEN_SIZE = 128 # lstm 隐藏层

def getBadCases(choose_model):
    badcases_contents = []
    badcases_scores = []
    badcases_true_labels = []
    badcases_pred_labels = []

    x, y, vocabulary, vocabulary_inv, labelToindex, _, labelNumdict = data_processing.load_input_data(MAXLENGTH)
    # word2vec预训练权重
    weight_array = pickle.load(open(os.path.join(DATA_PATH, 'weight_array'), 'rb'))

    # 选择test的模型
    if choose_model == 'TextCNN':
        model = TextCNN(1, KERNEL_NUM, len(vocabulary), EMBEDDING_DIM, len(labelToindex))
    elif choose_model == 'BiLSTM':
        model = BiLSTM(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex))
    elif choose_model == 'TextCNN_BN':
        model = TextCNN_BN(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array=None)
    elif choose_model == 'BiLSTM_b':
        model = BiLSTM_b(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex), MAXLENGTH)
    elif choose_model == 'CNN_BiLSTM_a':
        model = CNN_BiLSTM_a(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, HIDDEN_SIZE, len(labelToindex), MAXLENGTH)
    elif choose_model == 'CNN_with_pretrained_embedding':
        model = TextCNN_BN_with_pretrained_embed(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)
    elif choose_model == 'TextRCNN':
        model = TextRCNN(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, HIDDEN_SIZE, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)
    elif choose_model == 'TextCNN_multi_channel':
        model = TextCNN_multi_channel(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, choose_model + '_201807110957.pkl'))) # 日期要变
    if use_cuda:
        model = model.cuda()
    print("Model loaded!")

    # 所有样本
    all_samples = pd.read_csv(os.path.join(DATA_PATH, 'all_labeled_datas.csv'))
    all_samples_contents = all_samples['content']
    all_samples_scores = all_samples['score']
    all_samples_labels = all_samples['label']

    all_samples_pro_contents = []
    all_samples_pro_scores = []
    all_samples_pro_labels = []

    for content, score, label in zip(all_samples_contents, all_samples_scores, all_samples_labels):
        punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』 ]")
        digit = re.compile(u"[0-9]")
        number = re.compile(u"[a-zA-Z]")

        content = punctuation.sub('', content)
        content = digit.sub("", content)
        content = number.sub("", content)
        if content != '':
            all_samples_pro_contents.append(content)
            all_samples_pro_scores.append(score)
            all_samples_pro_labels.append(label)

    all_pro_seg_contents = []
    all_pro_seg_scores = []
    all_pro_seg_labels = []
    sentenceToindex = {}
    for content, score, label in zip(all_samples_pro_contents, all_samples_pro_scores, all_samples_pro_labels):
        seg_content = jieba.cut(content)
        seg_con = []
        for word in seg_content:
            if word not in data_processing.get_stop_words().keys() and word in vocabulary.keys():
                seg_con.append(word)

        # 文本去重
        tmpSentence = ''.join(seg_con)
        if tmpSentence != '':
            if tmpSentence in sentenceToindex:
                continue
            else:
                sentenceToindex[tmpSentence] = len(sentenceToindex)

            all_pro_seg_contents.append(seg_con)
            all_pro_seg_scores.append(score)
            all_pro_seg_labels.append(label)

    for i, ct in enumerate(all_pro_seg_contents):
        ct_pad = data_processing.pad_sentences([ct], MAXLENGTH)
        input_x, input_y = data_processing.build_input_data(ct_pad, all_pro_seg_labels[i], vocabulary)

        input_x = Variable(torch.LongTensor(input_x))
        input_y = Variable(torch.LongTensor(input_y))
        if use_cuda:
            input_x = input_x.cuda()
            input_y = input_y.cuda()

        model_out = model(input_x)
        _, pre_y = torch.max(model_out, 1)

        if pre_y.item() != input_y.item():
            badcases_contents.append(' '.join(all_pro_seg_contents[i]))
            badcases_scores.append(all_pro_seg_scores[i])
            badcases_true_labels.append(all_pro_seg_labels[i])
            badcases_pred_labels.append(pre_y.item())

    dataframe = pd.DataFrame({"content": badcases_contents, "user_score": badcases_scores, "true_label": badcases_true_labels,
                              "pred_label": badcases_pred_labels})
    dataframe.to_csv(os.path.join(DATA_PATH, 'badcases.csv'), index=False, sep=',')
    print("Badcases done!")


if __name__ == '__main__':
    print("Bad cases!")
    getBadCases(choose_model='TextCNN_BN')