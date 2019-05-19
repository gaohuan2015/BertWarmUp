import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

raw_text = open('word2vec/data.txt', encoding='utf-8').\
    read().split()

vocab = set(raw_text)

word_to_ix = {word:i for i, word in enumerate(vocab)} 

data = []
for i in range(2,len(raw_text)-2):
    context = raw_text[i]
    target = [raw_text[i-2],raw_text[i-1],
                raw_text[i+1],raw_text[i+2]]

    data.append((context,target))
print(data[:5])

# SkipGram网络
class SkipGram(nn.Module):
    def __init__(self, n_word, n_dim):
        super(SkipGram,self).__init__()
        self.embeddings = nn.Embedding(n_word,n_dim)
        self.linear1 = nn.Linear(25,128)
        self.linear2 = nn.Linear(128,n_word)

    def forward(self,inputs):
        embeds = self.embeddings(inputs).view(4,-1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out,dim=1)

        return log_probs

model = SkipGram(len(word_to_ix),100)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


# 训练过程
for epoch in range(10):
    total_loss = 0
    for context, target in data:
        context =  torch.LongTensor([word_to_ix[context]])
        target = torch.LongTensor([word_to_ix[i] for i in target])

        out = model(context)
        loss = loss_function(out, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(total_loss)
    print('loss: {:.6f}'.format(total_loss / len(data)))
print('losses:',losses)
# 数据准备


