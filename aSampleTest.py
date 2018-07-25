# -*- coding: utf-8 -*-
"""
    测试样本
"""
import os
import jieba
import re
import pickle

import torch
from torch.autograd import Variable

from data_process import data_processing
from TextCNN import TextCNN
from BiLSTM import BiLSTM
from TextCNN_BN import TextCNN_BN, TextCNN_multi_channel
from BiLSTM_b import BiLSTM_b
from CNN_BiLSTM_Concat import CNN_BiLSTM_a
from BiGRU import BiGRU
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

def aSampleTest(choose_model):
    x, y, vocabulary, vocabulary_inv, labelToindex, sentenceToindex, labelNumdict = data_processing.load_input_data(MAXLENGTH)
    # word2vec预训练权重
    weight_array = pickle.load(open(os.path.join(DATA_PATH, 'weight_array'), 'rb'))

    test_sample_x = '价格公正，物流很快，但有些污垢！'
    test_sample_y = 1
    test_sample_seg = []

    # 去除标点符号、数字及字母
    punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』 ＾┻]")
    digit = re.compile(u"[0-9]")
    number = re.compile(u"[a-zA-Z]")

    test_sample_x = punctuation.sub("", test_sample_x)
    test_sample_x = digit.sub("", test_sample_x)
    test_sample_x = number.sub("", test_sample_x)

    for word in jieba.cut(test_sample_x):
        if word not in data_processing.get_stop_words().keys() and word in vocabulary.keys():
            test_sample_seg.append(word)
    test_sample_seg_pad = data_processing.pad_sentences([test_sample_seg], MAXLENGTH)
    test_x, test_y = data_processing.build_input_data(test_sample_seg_pad, test_sample_y, vocabulary)

    test_x = Variable(torch.LongTensor(test_x))
    test_y = Variable(torch.LongTensor(test_y))
    if use_cuda:
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    # 选择test的模型
    if choose_model == 'TextCNN':
        model = TextCNN(1, KERNEL_NUM, len(vocabulary), EMBEDDING_DIM, len(labelToindex))
    elif choose_model == 'BiLSTM':
        model = BiLSTM(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex))
    elif choose_model == 'TextCNN_BN':
        model = TextCNN_BN(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH)
    elif choose_model == 'BiLSTM_b':
        model = BiLSTM_b(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex), MAXLENGTH)
    elif choose_model == 'CNN_BiLSTM_a':
        model = CNN_BiLSTM_a(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, HIDDEN_SIZE, len(labelToindex), MAXLENGTH)
    elif choose_model == 'BiGRU':
        model = BiGRU(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex), MAXLENGTH)
    elif choose_model == 'CNN_with_pretrained_embedding':
        model = TextCNN_BN_with_pretrained_embed(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)
    elif choose_model == 'TextRCNN':
        model = TextRCNN(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, HIDDEN_SIZE, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)
    elif choose_model == 'TextCNN_multi_channel':
        model = TextCNN_multi_channel(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)

    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, choose_model + '_201807102300.pkl'))) # 日期要变
    if use_cuda:
        model = model.cuda()

    model_out = model(test_x) # (1, 3)
    _, pre_y = torch.max(model_out, 1)
    print("预测的标签为：", pre_y.item())

if __name__ == '__main__':
    print("Test model...")
    aSampleTest(choose_model='TextCNN_multi_channel')