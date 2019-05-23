import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from io import open

torch.manual_seed(1)
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
raw_sentence = open('LETA-lv-en/leta.en',encoding='utf-8').\
        read().strip().split()

vocab = set(raw_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
data= []
#建立一个元组列表。每个元组是([ word_i-2, word_i-1 ], target word)的形式
data = [([raw_sentence[i], raw_sentence[i + 1]], raw_sentence[i + 2])
            for i in range(len(raw_sentence) - 2)]
# 一开始输出前三个，也可以输出目标词为第一个词，前两个词为句末的两个词，即([raw_sentence[i-2], raw_sentence[i - 1]], raw_sentence[i])
print(data[:3])


'''
建立元组列表的另一种形式如下
data= []
for i in range(2,len(raw_sentence)- 2):
  #建立一个元组列表。每个元组是([ word_i-2, word_i-1 ], target word)的形式
    context= [raw_sentence[i-2],raw_sentence[i-1]]
    target = raw_sentence[i]
    data.append((context,target))
# print the first 3, just so you can see what they look like
print(data[:3])
'''

class NNLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 512)
        self.linear2 = nn.Linear(512, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NNLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    print('epoch{}'.format(epoch))
    total_loss = 0
    for context, target in data:
        # 准备要传递到模型中的输入（即，将单词转换成整数索引，并用张量包起来）
        context = Variable(torch.tensor([word_to_ix[w] for w in context], dtype=torch.long))
        target = Variable(torch.tensor([word_to_ix[target]], dtype=torch.long))
        #  运行向前传递，在下一个单词上获得对数概率
        log_probs = model(context)
        #  计算损失函数。（同样地，torch希望目标词包含在张量中）
        loss = loss_function(log_probs, target)
        # 通过调用item()，从1元素张量得到总损失
        total_loss += loss.item()
        #  梯度归0
        optimizer.zero_grad()
        # 进行反向传递并更新梯度
        loss.backward()
        optimizer.step()
    print('loss: {:.4f}'.format(total_loss / len(data)))

    losses.append(total_loss)
print('losses:',losses)
# 损失随着训练数据的增加而减少