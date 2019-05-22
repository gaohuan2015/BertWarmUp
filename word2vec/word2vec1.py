import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import heapq



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



def context_to_tensor(context, idx_dict):

    context_idx = [idx_dict[word]for word in context ]
    # print(context_idx)

    return ((context_idx))


def getmax4(num_list,topk=4):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    max_num_index=map(num_list.index, heapq.nlargest(topk,num_list))
    min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
    print ('max_num_index:',max_num_index)
    print ('min_num_index:',min_num_index)
mod=0
if mod==0:
    word = open('leta.en', 'r', encoding='utf-8')
    word2 = open('leta2.en', 'r', encoding='utf-8')
    word = word.read().split()
    word2 = word2.read().split()
    word1 = set(word)
    word3 = set(word2)
    word_to_idx = {word: idx for idx, word in enumerate(word3)}
    print(word_to_idx)
    data = list()
    model = SKT(len(word3), yujing, qianru, hidden)
    data = list()
    for i in range(2, len(word1) - 2):
        # Context, target
        bow = (word[i], [word[i - 2], word[i - 1], word[i + 1], word[i + 2]])
        data.append(bow)
    # print(data)
    # print(len(word))
    print(model)
    optimizer = opt.Adam(model.parameters(), lr=jindu)
    loss_function = nn.NLLLoss()
    if torch.cuda.is_available():
        MyModel = model.cuda()
    for e in range(step):
        total_loss = torch.FloatTensor([0])
        relly = 0
        i = 0
        for bag in data:
            model.zero_grad()
            context_data = torch.LongTensor([word_to_idx[bag[0]]]).cuda()
            target_data = (context_to_tensor(bag[1], word_to_idx))
            # print(target_data)
            loss = 0
            i += 1
            # print(target_data[0],target_data[1],target_data[2],target_data[3])
            # print(bag[0], word_to_idx)
            # target_data = context_to_tensor(bag[1], word_to_idx).cuda()
            target_length=len(target_data)
            prediction = MyModel(context_data)
            for j in range(target_length):
                # print(target_data[j])
                # print(prediction,torch.LongTensor(list(target_data[j])))
                target_data1=Variable(torch.LongTensor([target_data[j]])).cuda()
                # print(target_data1)
                # print(prediction,target_data1)
                loss += loss_function(prediction, target_data1)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    torch.save(model, "word2vec1.pth")
    print('Step: {} | Loss: {}'.format(e, total_loss.numpy()))

    word = open('leta1.en', 'r', encoding='utf-8')
    word = word.read().split()
    word1 = set(word)
    # print(word_to_idx)
    data = list()
    for i in range(2, len(word1) - 2):
        # Context, target
        bow = (word[i], [word[i - 2], word[i - 1], word[i + 1], word[i + 2]])
        data.append(bow)
    # print(data)
    # print(len(word))
    # print(model)
    optimizer = opt.Adam(model.parameters(), lr=jindu)
    loss_function = nn.NLLLoss()
    if torch.cuda.is_available():
        MyModel = model.cuda()

    total_loss = torch.FloatTensor([0])
    relly = 0
    i = 0
    for bag in data:
            model.zero_grad()
            context_data = torch.LongTensor([word_to_idx[bag[0]]]).cuda()
            target_data = (context_to_tensor(bag[1], word_to_idx))
            # print(target_data)
            loss = 0
            i += 1
            # print(target_data[0],target_data[1],target_data[2],target_data[3])
            # print(bag[0], word_to_idx)
            # target_data = context_to_tensor(bag[1], word_to_idx).cuda()
            target_length = len(target_data)
            prediction = MyModel(context_data)
            for j in range(target_length):
                # print(target_data[j])
                # print(prediction,torch.LongTensor(list(target_data[j])))
                target_data1 = Variable(torch.LongTensor([target_data[j]])).cuda()
                # print(target_data1)
                # print(prediction,target_data1)
                loss += loss_function(prediction, target_data1)
    total_loss += loss.data
    print('Loss: {}'.format( total_loss.numpy()))