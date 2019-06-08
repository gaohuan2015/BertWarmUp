import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from io import open

class Lang:
    def __init__(self):
        self.word2index = {} #字典 词到索引的映射
        self.index2word = {} #字典 索引到词的映射
        self.index = 0

    def addWord(self,word):
        if word not in self.word2index: #如果词到索引的映射字典中不包含该词,则添加
            self.word2index[word] = self.index
            self.index2word[self.index] = word #创建索引到词的映射
            self.index +=1

    def __len__(self):
        return len(self.word2index) #词到索引映射的字典大小

class Corpus:
    def __init__(self):
        self.lang = Lang()

    def getData(self,batch_size=20):
        with open('data/train.txt','r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.lang.addWord(word)

        # 对文件做Tokenize
        ids = torch.LongTensor(tokens)
        token = 0
        with open('data/train.txt', 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.lang.word2index[word]
                    token +=1

        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)

# RNN based language model
class RNNLanguageModel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layers):
       super(RNNLanguageModel, self).__init__()
       # 嵌入层 one-hot形式(vocab_size,1) -> (embed_size,1)
       self.embed = nn.Embedding(vocab_size,embed_size)
       # LSTM单元/循环单元
       self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
       # 输出层的全联接操作
       self.linear = nn.Linear(hidden_size,vocab_size)

    def forward(self, input, hidden):
        embedded = self.embed(input)
        output, (h_n,c_n) = self.lstm(embedded, hidden)
        # Reshape output to (batch_size*sequence_length, hidden_size)
        # 每个时间步骤上LSTM单元都会有一个输出，batch_size个样本并行计算(每个样本/序列长度一致)  out (batch_size,sequence_length,hidden_size)
        # 把LSTM的输出结果变更为(batch_size*sequence_length, hidden_size)的维度
        output = output.reshape(output.size(0)*output.size(1),output.size(2))

        output = self.linear(output)  #(batch_size*sequence_length, hidden_size)->(batch_size*sequence_length, vacab_size)
        return output, (h_n,c_n)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 128
hidden_size = 1024  # 使用RNN变种LSTM单元
num_layers = 1      # 环单元/LSTM单元的层数
batch_size = 20
seq_length = 100    # 一个样本/序列长度

corpus = Corpus()
ids = corpus.getData(batch_size=20)
vocab_size = len(corpus.lang)
num_batches = ids.size(1) //  seq_length

model = RNNLanguageModel(vocab_size, embed_size, hidden_size, 1).to(device)
criterion = nn.CrossEntropyLoss()    #交叉熵损失
#使用Adam优化方法 最小化损失 优化更新模型参数
optimizer = optim.Adam(model.parameters(), lr = 0.002)

def detach(states):
    return [state.detach() for state in states]

# Train the model
for epoch in range(5):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers,batch_size,hidden_size).to(device),
            torch.zeros(num_layers,batch_size,hidden_size).to(device))

    for i in range(0, ids.size(1)-seq_length, seq_length):
        # 获取一个mini batch的输入和输出(标签)
        inputs = ids[:,i:i+seq_length].to(device)
        # 输出相对输入错一位，往后顺延一个单词
        targets = ids[:,(i+1):(i+1)+seq_length].to(device)

        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1)//seq_length
        if step % 200 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'.format(epoch+1, 5, step, num_batches, loss.item()))

# Test the model
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                torch.zeros(num_layers, 1, hidden_size).to(device))

        # 随机选择一个词作为输入
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(1000):
            output, state = model(input, state)

            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            input.fill_(word_id)

            word = corpus.lang.index2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 200 == 0:

                print('Sampled [{}/{}] words and save to {}'.format(i+1, 1000, 'sample.txt'))

torch.save(model.state_dict(), 'model.ckpt')