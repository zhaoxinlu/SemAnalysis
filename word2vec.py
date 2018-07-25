# -*-  coding: utf-8 -*-
from gensim.models import Word2Vec
import multiprocessing
import os
import numpy as np
import pickle

from data_process.data_processing import *

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

def w2v_train():
    seg_contents, all_labels, _, _, _ = get_contents_seg(DATA_PATH)

    print("Training w2v...")
    w2v_model = Word2Vec(seg_contents, size=128, min_count=2, workers=multiprocessing.cpu_count(), sg=0)
    w2v_model.save(os.path.join(MODEL_PATH, 'content_128.model'))
    w2v_model.wv.save_word2vec_format(os.path.join(MODEL_PATH, 'content_128.vector'), binary=False)
    print("Model train done!")

def save_weight_array(w2v_model_file):
    wordToindex = {'<PAD>': 0, '<UNK>': 1}
    weight_array = []
    with open(w2v_model_file, 'r', encoding='utf-8') as f_emb:
        for line in f_emb:
            line = line.strip().split()
            word = line[0]
            word_embed = len(line)-1
            if word_embed != 128:
                continue

            wordToindex[word] = len(wordToindex)
            weight_array.append(line[1:])
    indexToword = {v: k for k, v in wordToindex.items()}
    print(len(indexToword))
    weight_array = [np.random.uniform(-0.1, 0.1, 128).tolist()]+[np.random.uniform(-0.1, 0.1, 128).tolist()]+weight_array
    weight_array = np.array(weight_array)
    print(weight_array.shape)

    f_w2i = open(os.path.join(DATA_PATH, 'wordToindex'), 'wb')
    f_i2w = open(os.path.join(DATA_PATH, 'indexToword'), 'wb')
    pickle.dump(wordToindex, f_w2i, 1)
    pickle.dump(indexToword, f_i2w, 1)
    pickle.dump(weight_array, open(os.path.join(DATA_PATH, 'weight_array'), 'wb'), protocol=4)
    print("Save weight array done!")

if __name__ == '__main__':
    print("Hello World!")
    w2v_train()
    save_weight_array(os.path.join(MODEL_PATH, 'content_128.vector'))