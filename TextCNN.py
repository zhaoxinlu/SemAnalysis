# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class TextCNN(nn.Module):
    def __init__(self, in_channels, out_channels, vocab_size, embedding_dim, num_classes):
        super(TextCNN, self).__init__()

        self.in_channels = in_channels # 单通道，单个embedding层
        self.out_channels = out_channels
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, embedding_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (4, embedding_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (5, embedding_dim), padding=(4, 0))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(out_channels*3, num_classes)

    def forward(self, inputs):
        #print("Model input size is:", inputs.size())
        embed = self.embedding(inputs)
        #print("Embeded size is: ", embed.size()) # (batch, maxLength, embed_dim)
        x = embed.unsqueeze(1) # (batch, in_channels, maxLength, embed_dim)
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # x[0-2]: (batch, out_channels)
        cnn_x = torch.cat(x, 1) # (batch, out_channels*3)
        #print("CNN_X out size is: ", cnn_x.size())
        cnn_x = self.dropout(cnn_x)
        output = self.fc(cnn_x)
        #print("Output size is: ", output.size())
        return output