import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_

class Dictionary(object):
    '''
    构建word2id,id2word两个字典
    '''
    def __init__(self):
        self.word2idx = {} #字典 词到索引的映射
        self.idx2word = {} #字典  索引到词的映射
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx: #如果词到索引的映射字典中 不包含该词 则添加
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word #同时创建索引到词的映射
            self.idx += 1

    def __len__(self):
        return len(self.word2idx) #词到索引映射的字典大小


class Corpus(object):
    '''
    基于训练语料，构建字典(word2id,id2word)
    '''
    def __init__(self):
        self.dictionary = Dictionary() #创建字典类对象

    def get_data(self, path, batch_size=20):
        # 添加词到字典
        with open(path, 'r',encoding='utf8') as f:#读取文件
            tokens = 0
            for line in f:  #遍历文件中的每一行
                words = line.split() + ['<eos>'] #以空格分隔 返回列表 并添加一个结束符<eos>
                tokens += len(words)
                for word in words: #将每个单词添加到字典中
                    self.dictionary.add_word(word)

        # 对文件做Tokenize
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r',encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        return ids.view(batch_size, -1)
# 有gpu的情况下使用gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# 超参数的设定
embed_size = 128    # 词嵌入的维度
hidden_size = 1024  # 使用RNN变种LSTM单元   LSTM的hidden size
num_layers = 1      #循环单元/LSTM单元的层数
num_epochs = 5      # 迭代轮次
num_samples = 1000  # 测试语言模型生成句子时的样本数
batch_size = 20     # 一批样本的数量
seq_length = 30     # 一个样本/序列长度
learning_rate = 0.002 # 学习率
# 加载数据集
corpus = Corpus()
ids = corpus.get_data('data/DATA.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length
# RNN语言模型
class RNNLM(nn.Module): #RNNLM类继承nn.Module类
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        #嵌入层 one-hot形式(vocab_size,1) -> (embed_size,1)
        self.embed = nn.Embedding(vocab_size, embed_size)
        #LSTM单元/循环单元
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        #输出层的全联接操作
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # 词嵌入
        x = self.embed(x)

        # LSTM前向运算
        out,(h,c) = self.lstm(x,h)

        # 每个时间步骤上LSTM单元都会有一个输出，batch_size个样本并行计算(每个样本/序列长度一致)  out (batch_size,sequence_length,hidden_size)
        # 把LSTM的输出结果变更为(batch_size*sequence_length, hidden_size)的维度
        out = out.reshape(out.size(0)*out.size(1),out.size(2))
        # 全连接
        out = self.linear(out) #(batch_size*sequence_length, hidden_size)->(batch_size*sequence_length, vacab_size)

        return out,(h,c)

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# 损失构建与优化
criterion = nn.CrossEntropyLoss() #交叉熵损失
#使用Adam优化方法 最小化损失 优化更新模型参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 反向传播过程“截断”(不复制gradient)
def detach(states):
    return [state.detach() for state in states]

# 训练模型
for epoch in range(num_epochs):
    # 初始化为0
    states = (torch.zeros(num_layers,batch_size,hidden_size).to(device),
             torch.zeros(num_layers,batch_size,hidden_size).to(device))

    for i in range(0, ids.size(1) - seq_length, seq_length):
        # 获取一个mini batch的输入和输出(标签)
        inputs = ids[:,i:i+seq_length].to(device)
        targets = ids[:,(i+1):(i+1)+seq_length].to(device) # 输出相对输入错一位，往后顺延一个单词

        # 前向运算
        states = detach(states)
        outputs,states = model(inputs,states)
        loss = criterion(outputs,targets.reshape(-1))

        # 反向传播与优化
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(),0.5)

        step = (i+1) // seq_length
        if step % 20 == 0:
            print ('epoch: {}, Loss: {:.6f}'
                   .format(epoch+1, loss.item()))
            print('-'*30)

# 测试语言模型
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # 初始化为0
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # 随机选择一个词作为输入
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # 从输入词开始，基于语言模型前推计算
            output, state = model(input, state)

            # 做预测
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # 填充预估结果（为下一次预估储备输入数据）
            input.fill_(word_id)

            # 写出输出结果
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('生成了 [{}/{}] 个词，存储到 {}'.format(i+1, num_samples, 'sample.txt'))

# 存储模型的保存点(checkpoints)
torch.save(model.state_dict(), 'model.ckpt')