# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

use_cuda = torch.cuda.is_available()

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, bidirectional=True)

    def forward(self, input):
        return self.gru(input)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0., std=stdv)

        self.attn = nn.Linear(self.hidden_size * 4, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        # hidden :b,h*2
        # encoder_outputs : s,b,h*2
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)#s,b,h*2->b,s,h*2
        encoder_outputs = encoder_outputs.transpose(0, 1)#b,s,h*2
        energy = self.attn(torch.cat((H,encoder_outputs), 2))#b,s,h*4->b,s,h
        energy = energy.transpose(2, 1)#b,h,s
        v = self.v.repeat(batch_size, 1).unsqueeze(1)#b,1,h
        energy = torch.bmm(v, energy).squeeze()#b,s
        attention_weight = F.softmax(energy).unsqueeze(1)#b,1,s
        return attention_weight

# 定义模型
class Attention_RNN_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, weight_array):
        super(Attention_RNN_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.embed = nn.Embedding(vocab_size, embedding_dim)  # b,S,E->s,b,e
        # 预训练权重
        self.embed.weight.data.copy_(torch.from_numpy(weight_array.astype(float)))
        self.dropemb = nn.Dropout(p=0.5)
        self.encoder = Encoder(self.embedding_dim, self.hidden_size)
        self.attention = Attention(self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc=nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, x):
        x = self.embed(x)#b,s,e
        x = self.dropemb(x) #(batch, maxLength, embedding_dim)
        x = x.permute(1, 0, 2)
        encoder_outputs, encoder_hidden = self.encoder(x)#s,b,h*2  #2,b,h
        encoder_hidden = torch.cat((encoder_hidden[-1], encoder_hidden[-2]), dim=1)#b,h*2
        decoder_hidden = encoder_hidden
        attention_weight = self.attention(decoder_hidden, encoder_outputs)#b,1,s
        linear_combination = attention_weight.bmm(encoder_outputs.transpose(0, 1)).squeeze(1)#b,h*2
        decoder_output=self.decoder(linear_combination)#,b,256
        #out = torch.cat((decoder_output, z), dim=1)
        out = F.softmax(self.fc(decoder_output), dim=1)
        return out