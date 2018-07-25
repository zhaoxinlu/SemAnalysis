# -*- coding: utf-8 -*-
"""
共有数据集样本数： 11438
整个数据集的标签分布情况： {bad: 5051, mid: 2485, good: 3902}
对比模型效果：
    没有预先训练词向量：
    BiLSTM：Train 63.33%; Test 64.83% -- 很失败的模型，弃掉
    BiLSTM_b: Train 84.08%; Test 82.50%
    TextCNN：Train 94.30%; Test 82.57% --严重过拟合了。。。
    TextCNN_BN：Train 93.15%; Test 81.80% --
    CNN_BiLSTM_a: Train 88.77%; Test 81.83%
    BiGRU: Train 88.22%; Test 79.59%

    预先训练词向量：-- min_count=2
    TextCNN_BN_pretrained_embed: Train 86.80%, Test 81.69%
    TextRCNN: Train 86.63%, Test 82.03%
    TextCNN_BN_multi_channel: Train 87.90%, Test 83.11% -- 双通道
    RNN_attention: Train 83.98%; Test 81.95%
"""
import logging
import time
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from data_process import data_processing
from TextCNN import TextCNN
from BiLSTM import BiLSTM
from TextCNN_BN import TextCNN_BN, TextCNN_multi_channel
from BiLSTM_b import BiLSTM_b
from CNN_BiLSTM_Concat import CNN_BiLSTM_a
from BiGRU import BiGRU
from TextCNN_BN_Pretrained_embed import TextCNN_BN_with_pretrained_embed
from TextRCNN import TextRCNN
from attention_rnn import Attention_RNN_model

use_cuda = torch.cuda.is_available()
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'new_data')

# 定义一些超参数
BATCH_SIZE = 16
EMBEDDING_DIM = 128
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
MAXLENGTH = 50
N_STEPS = 50
KERNEL_SIZES = [3, 4, 5]
KERNEL_NUM = 128 # 卷积核数量
HIDDEN_SIZE = 128 # lstm 隐藏层

def train_and_test(choose_model):
    # 写入日志文件
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    runTime = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    handler = logging.FileHandler('./logs/'+choose_model+'_'+runTime+'.log.txt')
    handler.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("************************************************************")

    x, y, vocabulary, vocabulary_inv, labelToindex, sentenceToindex, labelNumdict = data_processing.load_input_data(MAXLENGTH)
    logger.info("The number of samples is: {}".format(len(sentenceToindex)))
    logger.info("The distribution of the all dataset label(With: 0-bad, 1-mid, 2-good):{}".format(labelNumdict))
    # word2vec预训练权重
    weight_array = pickle.load(open(os.path.join(DATA_PATH, 'weight_array'), 'rb'))

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
    print("Train Sample's distribution: {}".format(data_processing.get_labelNumdict(train_y)))
    print("Test Sample's distribution: {}".format(data_processing.get_labelNumdict(test_y)))
    logger.info("Train Sample's distribution: {}".format(data_processing.get_labelNumdict(train_y)))
    logger.info("Test Sample's distribution: {}".format(data_processing.get_labelNumdict(test_y)))
    logger.info("Some hyperparameters with lr:{}, wd:{}, embed:{}".format(LEARNING_RATE, WEIGHT_DECAY, EMBEDDING_DIM))

    train_x = torch.LongTensor(train_x)
    test_x = torch.LongTensor(test_x)
    train_y = torch.LongTensor(train_y)
    test_y = torch.LongTensor(test_y)

    trainDataset = data_processing.JDataset(train_x, train_y)
    testDataset = data_processing.JDataset(test_x, test_y)
    trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    testDataLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)

    # 选择训练的模型
    if choose_model == 'TextCNN':
        model = TextCNN(1, KERNEL_NUM, len(vocabulary), EMBEDDING_DIM, len(labelToindex))
    elif choose_model == 'BiLSTM':
        model = BiLSTM(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex))
    elif choose_model == 'TextCNN_BN':
        model = TextCNN_BN(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array=None)
    elif choose_model == 'BiLSTM_b':
        model = BiLSTM_b(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex), MAXLENGTH, weight_array=weight_array)
    elif choose_model == 'CNN_BiLSTM_a':
        model = CNN_BiLSTM_a(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, HIDDEN_SIZE, len(labelToindex), MAXLENGTH, weight_array=weight_array)
    elif choose_model == 'BiGRU':
        model = BiGRU(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex), MAXLENGTH)
    elif choose_model == 'CNN_with_pretrained_embedding':
        model = TextCNN_BN_with_pretrained_embed(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)
    elif choose_model == 'TextRCNN':
        model = TextRCNN(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, HIDDEN_SIZE, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)
    elif choose_model == 'TextCNN_multi_channel':
        model = TextCNN_multi_channel(len(vocabulary), EMBEDDING_DIM, KERNEL_SIZES, KERNEL_NUM, len(labelToindex), MAXLENGTH, weight_array)
    elif choose_model == 'Attention_rnn':
        model = Attention_RNN_model(len(vocabulary), EMBEDDING_DIM, HIDDEN_SIZE, len(labelToindex), weight_array)
    # 打印模型信息
    print(model)
    logger.info(model)
    if use_cuda:
        model = model.cuda()

    # 损失函数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimzier = optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_acc = 0
    best_model = None
    for step in range(N_STEPS):
        model.train()
        train_loss = 0.0
        train_acc = 0
        for i, data in enumerate(trainDataLoader):
            tr_x, tr_y = data
            #print("Tr_X's size is: ", tr_x.size())
            #print("Tr_Y size: ", tr_y.size())
            if use_cuda:
                tr_x = Variable(tr_x).cuda()
                tr_y = Variable(tr_y).cuda()
            else:
                tr_x = Variable(tr_x)
                tr_y = Variable(tr_y)

            # forward
            out = model(tr_x)
            loss = criterion(out, tr_y)
            train_loss += loss.item() * len(tr_y)
            _, pre = torch.max(out, 1)
            #print("***", pre.size())
            #print(pre)
            num_acc = (pre==tr_y).sum()
            train_acc += num_acc.item()
            #print(train_acc)

            # backward
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()
            if (i+1) % 100 == 0:
                print('[{}/{}], train loss is: {:.6f}, train acc is: {:.6f}'.format(i, len(trainDataLoader),
                                                                                    train_loss/(i*BATCH_SIZE),
                                                                                    train_acc/(i*BATCH_SIZE)))

                logger.info('[{}/{}], train loss is: {:.6f}, train acc is: {:.6f}'.format(i, len(trainDataLoader),
                                                                                          train_loss/(i*BATCH_SIZE),
                                                                                          train_acc/(i*BATCH_SIZE)))

        print('Step:[{}], train loss is: {:.6f}, train acc is: {:.6f}'.format(step,
                                                                              train_loss/(len(trainDataLoader)*BATCH_SIZE),
                                                                              train_acc/(len(trainDataLoader)*BATCH_SIZE)))

        logger.info('Step:[{}], train loss is: {:.6f}, train acc is: {:.6f}'.format(step,
                                                                                    train_loss / (len(trainDataLoader) * BATCH_SIZE),
                                                                                    train_acc / (len(trainDataLoader) * BATCH_SIZE)))

        model.eval()
        eval_loss = 0
        eval_acc = 0
        for i, data in enumerate(testDataLoader):
            te_x, te_y = data
            if use_cuda:
                te_x = Variable(te_x).cuda()
                te_y = Variable(te_y).cuda()
            else:
                te_x = Variable(te_x)
                te_y = Variable(te_y)
            out = model(te_x)
            loss = criterion(out, te_y)
            eval_loss += loss.item() * len(te_y)
            _, pre = torch.max(out, 1)
            num_acc=(pre==te_y).sum()
            eval_acc += num_acc.item()
        print('test loss is: {:.6f}, test acc is: {:.6f}'.format(eval_loss/(len(testDataLoader)*BATCH_SIZE),
                                                                 eval_acc/(len(testDataLoader)*BATCH_SIZE)))

        logger.info('test loss is: {:.6f}, test acc is: {:.6f}'.format(eval_loss / (len(testDataLoader) * BATCH_SIZE),
                                                                       eval_acc / (len(testDataLoader) * BATCH_SIZE)))

        if best_acc < (eval_acc/(len(testDataLoader)*BATCH_SIZE)):
            best_acc = eval_acc/(len(testDataLoader)*BATCH_SIZE)
            best_model = model.state_dict()
            print('best acc is {:.6f}, best model is changed.'.format(best_acc))

            logger.info('best acc is {:.6f}, best model is changed.'.format(best_acc))

    logger.info("Best acc is: {}".format(best_acc))
    logger.info("************************************************************")

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, choose_model+'_'+runTime+'.pkl'))

if __name__ == '__main__':
    print("Hello, World!")
    #train_and_test(choose_model='TextCNN')
    #print("TextCNN model done!")
    #train_and_test(choose_model='BiLSTM')
    #print("BiLSTM model done!")
    #train_and_test(choose_model='TextCNN_BN')
    #print("TextCNN_BN model done!")
    #train_and_test(choose_model="BiLSTM_b")
    #print("BiLSTM_b model done!")
    #train_and_test(choose_model='CNN_BiLSTM_a')
    #print("CNN_BiLSTM_a model done!")
    #train_and_test(choose_model='BiGRU')
    #print("BiGRU model done!")
    #train_and_test(choose_model='CNN_with_pretrained_embedding')
    #print("Done!")
    #train_and_test(choose_model='TextRCNN')
    #print("TextRCNN Done!")
    #train_and_test(choose_model='TextCNN_multi_channel')
    #print("TextCNN_multi_channel Done!")
    train_and_test(choose_model='Attention_rnn')
    print("DOne!")