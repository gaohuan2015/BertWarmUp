import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from model import NGramLanguageModeler

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim,128)
        self.linear2 = nn.Linear(128,vocab_size)
    
    def forward(self,inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out,dim=1)
        return log_probs

CONTEXT_SIZE = 2
EMBEDDING_DIM = 521

criterion = nn.NLLLoss()
train_sentence = open('train.txt', encoding='utf-8').read().split()
train_vocab = set(train_sentence)
model = NGramLanguageModeler(len(train_vocab), EMBEDDING_DIM, CONTEXT_SIZE)
train_trigrams = [([train_sentence[i], train_sentence[i+1]], train_sentence[i+2]) for i in range(len(train_sentence)-2)]
optimizer = optim.SGD(model.parameters(), lr=0.01)
word_to_ix = {word: i for i, word in enumerate(train_vocab)}


# 训练模型
for epoch in range(10):
    loss_total = 0
    for context, target in train_trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        target_idxs = torch.tensor([word_to_ix[target]], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = criterion(log_probs, target_idxs)
        loss.backward()
        optimizer.step()
        loss_total += loss
    print('Epoch [{}/{}]:\tLoss:{:.4f}'.format(epoch+1, 10, loss_total))

# 测试数据集
with torch.no_grad():
    test_sentence = open('test.txt', encoding='utf-8').read().split()
    test_trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2]) for i in range(len(test_sentence)-2)]
    test_vocab = set(test_sentence)
    word_to_ix1 = {word: i for i, word in enumerate(test_vocab)}
    correct = 0
    total = 0
    for context, target in test_trigrams:
        context_idxs = torch.tensor([word_to_ix1[w] for w in context], dtype=torch.long)
        target_idxs = torch.tensor([word_to_ix1[target]], dtype=torch.long)
        outputs = model(context_idxs)
        _, predicted = torch.max(outputs, 0)
        correct += (predicted == target_idxs).sum().item()
        total += 1
    print('Accuracy of the model: {} %'.format(correct / total))
