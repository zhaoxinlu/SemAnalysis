# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class TextCNN_BN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_kernels, num_classes, maxLength, weight_array=None):
        super(TextCNN_BN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels
        self.num_classes = num_classes

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        if weight_array is not None:
            # 预训练权重
            self.embedding.weight.data.copy_(torch.from_numpy(weight_array.astype(float)))

        conv_blocks = []
        for ks in self.kernel_sizes:
            maxpool_ks = maxLength-ks+1
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
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.num_kernels*len(kernel_sizes), num_classes)

    def forward(self, inputs):
        embed = self.embedding(inputs) # (batch, maxLength, embedding_size)
        x = embed.transpose(1, 2)
        x_list = [conv(x) for conv in self.conv_blocks]
        # [(batch, num_kernels, 1)...]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        # (batch, num_kernels*len(kernel_sizes))
        out = F.dropout(out, p=0.5, training=self.training)
        fout = F.softmax(self.fc(out), dim=1)
        # (batch, num_classes)
        return fout

class TextCNN_multi_channel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_kernels, num_classes, maxLength, weight_array):
        super(TextCNN_multi_channel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels
        self.num_classes = num_classes
        self.weight_array = weight_array

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pre_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pre_embedding.weight.data.copy_(torch.from_numpy(self.weight_array.astype(float)))
        #self.pre_embedding.weight.requires_grad = False

        conv_blocks = []
        for ks in self.kernel_sizes:
            maxpool_ks = maxLength-ks+1
            conv = nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_dim*2, out_channels=self.num_kernels, kernel_size=ks),
                nn.BatchNorm1d(num_features=self.num_kernels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_ks)
            )
            if use_cuda:
                conv = conv.cuda()
            conv_blocks.append(conv)

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.num_kernels*len(kernel_sizes), num_classes)

    def forward(self, inputs):
        embed = self.embedding(inputs) # (batch, maxLength, embedding_size)
        x = embed.transpose(1, 2)
        pre_embed = self.pre_embedding(inputs)
        pre_x = pre_embed.transpose(1, 2)
        x = torch.cat((x, pre_x), 1) # (batch,2embedding_size, maxLength)
        #print("***", x.size())
        x = self.dropout(x)
        x_list = [conv(x) for conv in self.conv_blocks]
        # [(batch, num_kernels, 1)...]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        # (batch, num_kernels*len(kernel_sizes))
        out = F.dropout(out, p=0.5, training=self.training)
        fout = F.softmax(self.fc(out), dim=1)
        # (batch, num_classes)
        return fout