import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

CONTEXT_SIZE = 4
raw_text = open('data/sanguoyanyi.txt', encoding='utf-8').\
        read().strip().split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)

word_to_ix = {word:i for i, word in enumerate(vocab)}
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
data = []
for i in range(4,len(raw_text)-4):
    context = [raw_text[i-4],raw_text[i-3],raw_text[i-2],raw_text[i-1],raw_text[i+1],raw_text[i+2],raw_text[i+3],raw_text[i+4]]
    target = raw_text[i]
    data.append((context,target))
print(data[:9])

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW,self).__init__()
        self.embeddings = nn.Embedding(n_word,n_dim)
        self.linear1 = nn.Linear(2*context_size*n_dim,128)
        self.linear2 = nn.Linear(128,n_word)
    def forward(self,inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out,dim=1)
        return log_probs

model = CBOW(len(word_to_ix),10,CONTEXT_SIZE)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

for epoch in range(10):
    print('epoch{}'.format(epoch))
    print('*'*10)
    total_loss = 0
    for context, target in data:
        context = Variable(torch.LongTensor([word_to_ix[i] for i in context]))
        target = Variable(torch.LongTensor([word_to_ix[target]]))
        out = model(context)
        loss = loss_function(out,target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss: {:.6f}'.format(total_loss / len(data)))
