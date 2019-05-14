import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from setData import setData
from model import CBOW


window = 2          #单词左右单词个数
EMBED_SIZE=50       
LEARN_RATE=0.02   #学习率
BATCH_SIZE = 100     #批训练参数
DEVICE = "cuda"     # "cpu" GPU跑还是CPU
TRAIN_NUM = 10     #训练次数


corpus = open('leta.en',encoding='utf8').read().split()
words = set(corpus)#转换为集合去除重复的单词
wordDic = {word: id for id, word in enumerate(words)}   #生成单词字典，一个单词对应一个ID号
# enumerate()     将单词集合组合为一个索引序列
data = []
label = []
for i in range(window, len(corpus) - window):
    data.append(np.array([wordDic[corpus[i+j]] for j in range(-window,window+1) if j], dtype=np.int64))
    label.append(wordDic[corpus[i]])

# 转为torch数据集
dataset = setData(data, label)

model = CBOW(vocabulary_size=len(words), embedding_size=EMBED_SIZE)
model = model.to(DEVICE)
# 损失
loss_func = nn.CrossEntropyLoss()
# 参数优化
optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

# 批训练
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

#开始训练
for e in range(TRAIN_NUM):
    #total_loss = torch.FloatTensor([0])
    batch_generator = generate_batches(dataset, batch_size=BATCH_SIZE, device=DEVICE)
    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        y_pred = model(x_in=batch_dict['x_data'])
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss.backward()
        optimizer.step()
        print('\rEpoch: %d | Loss: %f.4' % (e, loss.data) , end="")
    print("")

print("TRAIN END")

def get_closest(target_word, word_to_idx, embeddings, n=5):
    word_embedding = embeddings[word_to_idx[target_word.lower()]]
    
    distances = []
    for word, index in word_to_idx.items():
        distances.append((word, torch.dist(word_embedding, embeddings[index])))
    results = sorted(distances, key=lambda x: x[1])[0:n+2]
    return results


def pretty_print(results):
    for item in results:
        print ("...[%.2f] - %s"%(item[1], item[0]))

embeddings = model.embedding.weight.data

pretty_print(get_closest('together', wordDic, embeddings, n=5))