# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class TextCNN_BN_with_pretrained_embed(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_kernels, num_classes, maxLength, weight_array):
        super(TextCNN_BN_with_pretrained_embed, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels
        self.num_classes = num_classes

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
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
        embed = self.dropout(embed)
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