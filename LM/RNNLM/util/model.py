import torch
import torch.nn as nn
# RNN model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out,(h,c) = self.lstm(x,h)  # LSTM前向运算
        out = out.reshape(out.size(0)*out.size(1),out.size(2))# 把LSTM的输出结果变更为维度reshape
        out = self.linear(out)
        return out,(h,c)