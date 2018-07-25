# -*- coding: utf-8 -*-
"""
    数据处理相关操作
"""
import os
import pandas as pd
import jieba
import re
import numpy as np
import pickle
from collections import Counter
import itertools
from torch.utils import data
from sklearn.model_selection import train_test_split

#DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datas')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'new_data')

def get_all_datas_csv(data_path):
    files = os.listdir(data_path)
    print("files:", files)
    all_contents = []
    all_scores = []
    for file in files:
        print(file)
        datas = pd.read_csv(os.path.join(data_path, file))
        contents = datas["content"]
        scores = datas["score"]
        for i in range(len(scores)):
            score_num = file.split('.')[0][-1]
            if scores[i] != score_num:
                scores[i] = score_num
        for content, score in zip(contents, scores):
            if type(content) != float: # 针对content数据缺失值nan问题
                all_contents.append(content)
                all_scores.append(score)

    dataframe = pd.DataFrame({"content": all_contents, "score": all_scores})
    dataframe.to_csv(os.path.join(data_path, "all_datas.csv"), index=False, sep=',')

def get_pro_contents_csv(data_path):
    datas = pd.read_csv(os.path.join(data_path, 'all_labeled_datas.csv'))
    contents = datas['content']
    scores = datas['score']
    labels = datas['label']
    pro_contents = []
    all_scores = []
    all_labels = []

    for content, score, label in zip(contents, scores, labels):
        # 去除标点符号、数字及字母
        punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』 ]")
        digit = re.compile(u"[0-9]")
        number = re.compile(u"[a-zA-Z]")

        content = punctuation.sub('', content)
        content = digit.sub("", content)
        content = number.sub("", content)
        if content != "":
            pro_contents.append(content)
            all_scores.append(score)
            all_labels.append(label)

    dataframe = pd.DataFrame({"content": pro_contents, "score": all_scores, "label": all_labels})
    dataframe.to_csv(os.path.join(data_path, "all_pro_labeled_datas.csv"), index=False, sep=',')#, encoding='utf8'
    print("Generate CSV datas done!")

def get_stop_words():
    """
        获取无用词表
    :return:
    """
    stop_words_file = os.path.join(DATA_PATH, 'stop_words.txt')
    stop_words = open(stop_words_file, 'r', encoding='utf8').readlines()
    stop_words = [sw.strip() for sw in stop_words]
    stop_words_dict = {}
    for sw in stop_words:
        if sw not in stop_words_dict:
            stop_words_dict[sw] = len(stop_words_dict)
    return stop_words_dict

def get_contents_seg(data_path):
    """
        分词，去无用词
    :param data_path:
    :return:
    """
    stop_words_dict = get_stop_words()

    datas = pd.read_csv(os.path.join(data_path, 'all_pro_labeled_datas.csv'))
    print("READ datas done!")
    contents = datas['content']
    scores = datas['score']
    labels = datas['label']
    seg_contents = []
    all_scores = []
    all_labels = []
    #labelToindex = {'非常不满意': 0, '较不满意': 1, '一般': 2, '较满意': 3, '非常满意': 4}
    labelToindex = {'差评': 0, '中评': 1, '好评': 2}
    indexTolabel = {v: k for k, v in labelToindex.items()}
    sentenceToindex = {}

    for content, score, label in zip(contents, scores, labels):
        # 去除标点符号、数字及字母
        punctuation = re.compile(u"[-~!@#$%^&*()_+`=\[\]\\\{\}\"|;':,./<>?·！@#￥%……&*（）——+【】、；‘：“”，。、《》？「『」』 ＾┻]")
        digit = re.compile(u"[0-9]")
        number = re.compile(u"[a-zA-Z]")

        content = punctuation.sub("", content)
        content = digit.sub("", content)
        content = number.sub("", content)

        seg_content = jieba.cut(content)
        seg_con = []
        for sc in seg_content:
            if sc not in stop_words_dict.keys():
                seg_con.append(sc)

        # 文本去重
        tmpSentence = ''.join(seg_con)
        if tmpSentence != '':
            if tmpSentence in sentenceToindex:
                continue
            else:
                sentenceToindex[tmpSentence] = len(sentenceToindex)

            seg_contents.append(seg_con)
            all_scores.append(score)
            all_labels.append(label)

    # 查看每一个类别的样本数
    labelNumdict = {}
    for lbe in all_labels:
        if lbe not in labelNumdict.keys():
            labelNumdict[lbe] = 1
        else:
            labelNumdict[lbe] += 1
    print("共有数据集样本数：", len(sentenceToindex))
    print("整个数据集的标签分布情况：", labelNumdict)

    return seg_contents, all_labels, labelToindex, sentenceToindex, labelNumdict

def pad_sentences(sentences, sequence_length, padding_word='<PAD>'):
    #sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv

def build_input_data(sentences, labels, vocabulary):
    #x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    x = []
    for sentence in sentences:
        x_a = []
        for word in sentence:
            if word in vocabulary.keys():
                x_a.append(vocabulary[word])
            else:
                x_a.append(vocabulary['<UNK>'])
        x.append(x_a)
    x = np.array(x)
    y = np.array(labels)
    # y = y.argmax(axis=1)
    return [x, y]

def load_input_data(maxLength):
    seg_contents, labels, labelToindex, sentenceToindex, labelNumdict = get_contents_seg(DATA_PATH)
    pad_contents = pad_sentences(seg_contents, maxLength)
    #vocabulary, vocabulary_inv = build_vocab(pad_contents)
    vocabulary = pickle.load(open(os.path.join(DATA_PATH, 'wordToindex'), 'rb'))
    vocabulary_inv = pickle.load(open(os.path.join(DATA_PATH, 'indexToword'), 'rb'))
    x, y = build_input_data(pad_contents, labels, vocabulary)
    return x, y, vocabulary, vocabulary_inv, labelToindex, sentenceToindex, labelNumdict

def get_labelNumdict(LabelList):
    labelNumdict = {}
    for label in LabelList:
        if label not in labelNumdict.keys():
            labelNumdict[label] = 1
        else:
            labelNumdict[label] += 1
    return labelNumdict

class JDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

if __name__ == '__main__':
    # get_all_datas_csv(DATA_PATH)
    #get_pro_contents_csv(DATA_PATH)
    x, y, vocabulary, vocabulary_inv, labelToindex, sentenceToindex, labelNumdict = load_input_data(50)
    print(x[-1].shape)
    print(y[-1])
    #train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
    # 对比训练样本与测试样本标签分布情况
    #print("Train sample:", get_labelNumdict(train_y))
    #print("Test sample:", get_labelNumdict(test_y))
    #print(vocabulary)