# -*- coding: utf-8 -*-
"""
    Model Arch.
        Embedding -> BiLstm
        Concat(embedding, BiLstm_out) -> Conv+BN+MaxPool -> MLP
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class TextRCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, hidden_size, kernel_num, num_classes, maxLength, weight_array=None, n_layer=1):
        super(TextRCNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.hidden_size = hidden_size
        self.kernel_num = kernel_num
        self.num_classes = num_classes
        self.maxLength = maxLength
        self.n_layer = n_layer

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if weight_array is not None:
            # 预训练权重
            self.embedding.weight.data.copy_(torch.from_numpy(weight_array.astype(float)))

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.n_layer, dropout=0.2, batch_first=True, bidirectional=True)

        conv_blocks = []
        for ks in self.kernel_sizes:
            maxpool_ks = maxLength - ks + 1
            conv = nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_dim*2+self.embedding_dim, out_channels=self.kernel_num, kernel_size=ks),
                nn.BatchNorm1d(num_features=self.kernel_num),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_ks)
            )
            if use_cuda:
                conv = conv.cuda()
            conv_blocks.append(conv)

        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(
            nn.Linear(self.kernel_num*len(kernel_sizes), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes)
        )

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
        concat = torch.cat((bilstm_out, embed), 2) # (batch, maxLength, 2hidden_size+embed_dim)
        concat = self.dropout(concat)
        #print("Concat size: ", concat.size())
        concat = concat.transpose(1, 2)

        x_list = [conv(concat) for conv in self.conv_blocks]
        last_out = torch.cat(x_list, 2)
        #last_out = self.conv(concat) # ([batch, kernel_num, 1])
        #print("******", last_out.size())
        last_out = last_out.view(last_out.size(0), -1)
        last_out = self.dropout(last_out)
        output = F.softmax(self.fc(last_out), dim=1)
        #print("out size is: ", output.size())
        return output