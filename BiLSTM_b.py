# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class BiLSTM_b(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, maxLength, n_layer=1, weight_array=None):
        super(BiLSTM_b, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.maxLength = maxLength
        self.n_layer = n_layer

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if weight_array is not None:
            # 预训练权重
            self.embedding.weight.data.copy_(torch.from_numpy(weight_array.astype(float)))

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.n_layer, dropout=0.2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*2*self.maxLength, self.hidden_size*2),
            nn.BatchNorm1d(self.hidden_size*2),
            nn.ReLU(True),
            nn.Linear(self.hidden_size*2, self.num_classes)
        )

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(self.n_layer*2, inputs.size(0), self.hidden_size))
        context = Variable(torch.zeros(self.n_layer*2, inputs.size(0), self.hidden_size))
        if use_cuda:
            hidden = hidden.cuda()
            context = context.cuda()

        return (hidden, context)

    def forward(self, inputs):
        embed = self.embedding(inputs) # (batch, maxLength, embed_dim)
        embed = self.dropout(embed)
        hidden = self.init_hidden(inputs) #
        bilstm_out, hidden = self.lstm(embed, hidden) # bilstm_out: (batch, maxLength, 2hidden_size)
        #print("Output size is: ", bilstm_out.size())
        last_out = bilstm_out.contiguous().view(bilstm_out.size(0), -1)
        #print("******", last_out.size())
        output = F.softmax(self.fc(last_out), dim=1)
        #print("out size is: ", output.size())
        return output