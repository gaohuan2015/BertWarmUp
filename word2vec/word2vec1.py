import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import heapq
word = open('leta.en', 'r', encoding='utf-8')
word = word.read().split()
word1 = set(word)
word_to_idx = {word: idx for idx, word in enumerate(word1)}
data = list()
for i in range(2, len(word1) - 2):
    # Context, target
    bow = (word[i], [word[i - 2], word[i - 1], word[i + 1], word[i + 2]])
    data.append(bow)
print(data)


class SKT(nn.Module):
    def __init__(self, juzi_size, yujing_size, qianru_size, hidden_size):
        super(SKT, self).__init__()
        self.embed_layer = nn.Embedding(juzi_size, qianru_size)
        self.linear_1 = nn.Linear(qianru_size, hidden_size)
        self.linear_1_5 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, juzi_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        embeds = self.embed_layer(input_data).view((1, -1))
        output = F.relu(self.linear_1(embeds))
        output = F.relu(self.linear_1_5(output))
        output = self.log_softmax(self.linear_2(output))
        return output


yujing = 2
qianru = 32
hidden = 128
jindu = 0.0001
step = 30
print(len(word))

model = SKT(len(word1), yujing, qianru, hidden)
print(model)
optimizer = opt.Adam(model.parameters(), lr=jindu)
loss_function = nn.NLLLoss()
if torch.cuda.is_available():
    MyModel = model.cuda()


def context_to_tensor(context, idx_dict):

    context_idx = [idx_dict[context] ]
    # print(context_idx)

    return Variable(torch.LongTensor(context_idx))


def getmax4(num_list,topk=4):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
    min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
    print ('max_num_index:',max_num_index)
    print ('min_num_index:',min_num_index)

for e in range(step):
    total_loss = torch.FloatTensor([0])
    relly = 0
    i = 0
    for bag in data:
        loss = 0
        i += 1
        # print(bag[0], word_to_idx)
        # target_data = context_to_tensor(bag[1], word_to_idx).cuda()
        for j in range(4):
            context_data = torch.LongTensor([word_to_idx[bag[0]]]).cuda()
            target_data = (context_to_tensor(bag[1][j], word_to_idx)).cuda()
            # print(target_data)
            model.zero_grad()
            prediction = MyModel(context_data)
            loss = loss_function(prediction, target_data)
            loss.backward()
            optimizer.step()
        prediction1=prediction.cpu()
        prediction_numpy=prediction1.detach().numpy()
        # print(word_to_idx[bag[1][0]])
        # print(type(prediction_numpy))
        ans1=list()
        for i1 in bag[1]:
            ans1.append(float(prediction_numpy[0,word_to_idx[i1]]))
        # print(ans1)
        maxout=list()
        maxout.append(list(prediction_numpy[0,:]))
        maxout=maxout[0]
        # print(maxout)
        outm=list()
        for i in range(4):
            outm.append(max(maxout))
            maxout.remove([max(maxout)])
        print(ans1)
        print(outm)
        total_loss += loss.data
        # if (prediction[0,word_to_idx[bag[1]]]==torch.max(prediction)):
        #     relly+=1
        # else:
        #     relly+=0
    # Bookkeeping
    torch.save(model, "word2vec1.pth")
    # print(relly/i)
    print('Step: {} | Loss: {}'.format(e, total_loss.numpy()))