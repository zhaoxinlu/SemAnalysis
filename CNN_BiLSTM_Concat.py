# -*- coding: utf-8 -*-
"""
    Model Arch.
        embedding -> bilstm
        embedding -> CNN
        Concat(cnn_out, bilstm_out) -> Linear -> softmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class CNN_BiLSTM_a(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_kernels, hidden_size, num_classes, maxLength, n_layer=1, weight_array=None):
        super(CNN_BiLSTM_a, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.maxLength = maxLength
        self.n_layer = n_layer

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if weight_array is not None:
            # 预训练权重
            self.embedding.weight.data.copy_(torch.from_numpy(weight_array.astype(float)))

        conv_blocks = []
        for ks in self.kernel_sizes:
            maxpool_ks = maxLength - ks + 1
            conv = nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_kernels, kernel_size=ks),
                nn.BatchNorm1d(num_features=self.num_kernels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_ks)
            )
            if use_cuda:
                conv = conv.cuda()
            conv_blocks.append(conv)

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.conv_fc = nn.Linear(self.num_kernels*len(kernel_sizes), 100)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.n_layer, dropout=0.2, batch_first=True, bidirectional=True)
        self.lstm_fc = nn.Linear(self.hidden_size*2*self.maxLength, 100)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(100*2, self.num_classes)

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

        # cnn
        cnn_x = embed.transpose(1, 2)
        x_list = [conv(cnn_x) for conv in self.conv_blocks]
        # [(batch, num_kernels, 1)...]
        cnn_out = torch.cat(x_list, 2)
        cnn_out = cnn_out.view(cnn_out.size(0), -1) # (batch, num_kernels*len(kernel_sizes))
        cnn_out = self.dropout(cnn_out)
        cnn_fc_out = self.conv_fc(cnn_out) # (batch, 100)

        # bilstm
        hidden = self.init_hidden(inputs) #
        bilstm_out, hidden = self.lstm(embed, hidden) # bilstm_out: (batch, maxLength, 2hidden_size)
        #print("Output size is: ", bilstm_out.size())
        bilstm_out = bilstm_out.contiguous().view(bilstm_out.size(0), -1)
        #print("******", last_out.size())
        bilstm_out = self.dropout(bilstm_out)
        bilstm_fc_out = self.lstm_fc(bilstm_out) # (batch, 100)

        last_out = torch.cat((cnn_fc_out, bilstm_fc_out), 1)
        last_out = self.dropout(last_out)
        output = F.softmax(self.fc(last_out), dim=1)
        #print("out size is: ", output.size())
        return output