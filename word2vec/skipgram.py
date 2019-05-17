import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

#准备数据
#准备数据
CONTEXT_SIZE = 2
TARGET_SIZE=1
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
#数据处理
word_to_ix = {word: i for i, word in enumerate(vocab)}
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
data = []
for i in range(2, len(raw_text) - 2):
    target = raw_text[i]  # 目标词
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]] #上下文，各取2个词
    data.append((target,context))
print(data[:5])


#SkipGram模型
class SkipGram(nn.Module):
    def __init__(self, n_word, n_dim,target_size):
        super(SkipGram, self).__init__()
        #实例化
        self.embeddings = nn.Embedding(n_word, n_dim)  #embedding把输入的词变为向量
        self.linear1 = nn.Linear(target_size* n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)
#训练模型
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        #view(1,-1)是把字典转化到一行
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

#整个过程的实例化过程
model = SkipGram(len(word_to_ix), 100, TARGET_SIZE)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


for epoch in range(10):
    print('epoch{}'.format(epoch))
    #print('*' * 10)
    total_loss = 0 #总损失
    for target,context, in data:
        target = Variable(torch.LongTensor([word_to_ix[target]]))
        context = Variable(torch.LongTensor([word_to_ix[i] for i in context]))
        out = model(target)
        loss = loss_function(out, target) #计算损失
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward() #损失反向传播
        optimizer.step()

    print('loss: {:.6f}'.format(total_loss / len(data)))