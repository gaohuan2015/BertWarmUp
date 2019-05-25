import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
#CONTEXT_SIZE 表示我们希望由前面几个单词来预测这个单词，这里使用两个单词
CONTEXT_SIZE = 2 # 依据的单词数
EMBEDDING_DIM = 10 # 词向量的维度
#加载数据
test_sentence = open('data/DATA.txt', encoding='utf-8').\
        read().split()

##CONTEXT_SIZE 表示我们希望由前面几个单词来预测这个单词，这里使用两个单词，EMBEDDING_DIM 表示词嵌入的维度
trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2])
            for i in range(len(test_sentence)-2)]

# 总的数据量
len(trigram)

# 总的数据量
len(trigram)

#建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence) # 使用 set 将重复的元素去掉
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

#从上面可以看到每个词都对应一个数字，且这里的单词都各不相同
# 定义模型
class n_gram(nn.Module):
    def __init__(self, vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM):
        super(n_gram, self).__init__()

        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x):
        voc_embed = self.embed(x) # 得到词嵌入
        voc_embed = voc_embed.view(1, -1) # 将两个词向量拼在一起
        out = self.classify(voc_embed)
        return out

net = n_gram(len(word_to_idx))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)

for e in range(100):
    train_loss = 0
    for word, label in trigram: # 使用前 100 个作为训练集
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word])) # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # 前向传播
        out = net(word)
        loss = criterion(out, label)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch: {}, Loss: {:.6f}'.format(e+1, train_loss / len(trigram)))
        print('-'*30)
net = net.eval()
# 测试一下结果
word, label = trigram[24]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].item()
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))
