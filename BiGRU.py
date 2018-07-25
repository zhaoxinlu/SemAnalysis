# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class BiGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, maxLength, n_layer=1):
        super(BiGRU, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.maxLength = maxLength
        self.n_layer = n_layer

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.bigru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layer, dropout=0.2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_size*2*self.maxLength, num_classes)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, inputs):
        context = Variable(torch.zeros(self.n_layer*2, inputs.size(0), self.hidden_size))
        if use_cuda:
            context = context.cuda()

        return context

    def forward(self, inputs):
        embed = self.embedding(inputs) # (batch, maxLength, embed_dim)
        hidden = self.init_hidden(inputs) #
        #print("hidden size")
        bigru_out, hidden = self.bigru(embed, hidden) # bilstm_out: (batch, maxLength, 2hidden_size)
        #print("Output size is: ", bigru_out.size())
        last_out = bigru_out.contiguous().view(bigru_out.size(0), -1)
        #print("******", last_out.size())
        last_out = self.dropout(last_out)
        output = F.softmax(self.fc(last_out), dim=1)
        #print("out size is: ", output.size())
        return output