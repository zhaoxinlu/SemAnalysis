# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, n_layer=1):
        super(BiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.num_classes = num_classes

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.n_layer, dropout=0.2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, num_classes)

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
        hidden = self.init_hidden(inputs) #
        bilstm_out, hidden = self.lstm(embed, hidden) # bilstm_out: (batch, maxLength, 2hidden_size)
        #print("Output size is: ", bilstm_out.size())
        last_out = bilstm_out.transpose(0, 1)
        last_out = last_out[-1::]
        last_out = last_out.squeeze(0)
        #print("******", last_out.size())
        output = self.fc(last_out)
        #print("out size is: ", output.size())
        return output