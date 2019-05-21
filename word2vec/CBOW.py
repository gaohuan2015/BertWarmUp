import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from io import open
#准备数据
CONTEXT_SIZE = 3
raw_text = open('LETA-lv-en/leta.en',encoding='utf-8').\
        read().strip().split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
#数据处理
word_to_ix = {word: i for i, word in enumerate(vocab)}
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
data = []
for i in range(3, len(raw_text) - 3):
    context= [raw_text[i - 3],raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2],raw_text[i + 3]] #上下文，各取3个词
    target = raw_text[i] #目标词
    data.append((context, target))
print(data[:7]) #每次输出7个词

#CBOW模型
class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        #实例化
        self.embeddings = nn.Embedding(n_word, n_dim)  #embedding把输入的词变为向量
        self.linear1 = nn.Linear(2 * context_size * n_dim, 128)  #全连接层1实例化
        self.linear2 = nn.Linear(128, n_word)  #全连接层2实例化
#训练模型
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1)) #把输入的词向量化
        #view(1,-1)是把字典转化到一行
        out = F.relu(self.linear1(embeds))  #全连接层1，成一个一维向量，然后relu激活函数（0.5）为一个特征提取，进行筛选，把低于0.5的去掉。
        out = self.linear2(out)  #全连接层2与前面的筛选过后的大小一样
        log_probs = F.log_softmax(out, dim=1) #输出softmax层
        return log_probs

#整个过程的实例化过程
model = CBOW(len(word_to_ix), 100, CONTEXT_SIZE)
losses = []
loss_function = nn.NLLLoss() #损失函数即目标函数，这里使用了常用的交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.001) #SGD优化

#训练
for epoch in range(10):
    print('epoch{}'.format(epoch))
    #print('*' * 10)
    total_loss = 0 #总损失
    for context, target in data:
        context = Variable(torch.LongTensor([word_to_ix[i] for i in context]))
        target = Variable(torch.LongTensor([word_to_ix[target]]))
        out = model(context)
        loss = loss_function(out, target) #计算损失
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward() #损失反向传播
        optimizer.step()

    print('loss: {:.4f}'.format(total_loss / len(data)))