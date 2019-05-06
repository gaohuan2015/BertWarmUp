from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder.util import ConfigOp as cop

device=torch.device("cuda"if torch.cuda.is_available()else"cpu")

#编码器
class  EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # 指定embedding矩阵W的大小维度
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 指定gru单元的大小
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # 扁平化嵌入矩阵
        embedded = self.embedding(input).view(1, 1, -1)
        # print("embedded shape:",embedded.shape)
        output = embedded

        output, hidden = self.gru(output, hidden)
        return output, hidden

    #全0初始化隐层
    def initHidden(self):
        # 这个初始化维度可以
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def self__init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size,hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
         # 1行X列的shape做relu
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #output[0]应该是shape为（*，*）的矩阵
        output = self.softmax(self.out(output[0]))
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,
                dropout_p = 0.1, max_length=cop.MAX_LENGTH):
        super(AttnDecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        #输入向量的维度是10,隐层的长度是10,默认是一层GRU
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0],hidden[0]),1)),dim=1)
        # unsqueeze:在指定的轴上多增加一个维度
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0],attn_applied[0]),1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #print("output shape:",output.shape)
        #print("output[0]:",output[0])
        output = F.log_softmax(self.out(output[0]),dim=1)
        return output , hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)




