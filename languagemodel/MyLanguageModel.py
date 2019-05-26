import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
study = open('leta.en', encoding='utf-8')
study = study.read()
test_sentence = study.split()
print(test_sentence)
dit = open('leta2.en', encoding='utf-8')
dit = dit.read()
dit_sentence = dit.split()
ditvocab = set(dit_sentence)
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)

word_to_ix = {word: i for i, word in enumerate(ditvocab)}
ix_to_word = {i: word for i, word in enumerate(ditvocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        # 简单的查找表，存储固定字典和大小的嵌入。通常用于存储单词嵌入并使用索引来检索它们。
        # 模块的输入是一个索引列表，输出是相应的单词嵌入。
        # 初始化一个矩阵(vocab_size * embedding_dim)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # 取那个矩阵里面inputs所在的行
        embdeds = self.embeddings(inputs).view((1, -1))

        out = F.relu(self.linear1(embdeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs, self.embeddings


# model
model = NGramLanguageModeler(len(ditvocab), EMBEDDING_DIM, CONTEXT_SIZE).cuda()

losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(2000):
    total_loss = torch.Tensor([0]).cuda()
    for context, target in trigrams:
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs)).cuda()

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs, embedd = model(context_var)
        tagert = autograd.Variable(torch.LongTensor([word_to_ix[target]])).cuda()
        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, tagert)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
    if epoch % 100 == 0:
        print(total_loss[0])  # The loss decreased every iteration over the training data!
a = open('answ.txt', 'w', encoding='utf-8')
torch.save(model, 'language.pth')
a.write(str(ix_to_word) + '\n' + str(word_to_ix) + '\n')
for i in range(0, len(vocab)):
    lookup_tensor = torch.LongTensor([i]).cuda()
    a.write(ix_to_word[i] + str(embedd(autograd.Variable(lookup_tensor))) + '\n')
    # print(ix_to_word[i])
    # print(embedd(autograd.Variable(lookup_tensor)))
a.close()

study = open('leta1.en', encoding='utf-8')
study = study.read()
test_sentence = study.split()
print(test_sentence)
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)
total_loss = torch.Tensor([0]).cuda()
for context, target in trigrams:
    context_idxs = [word_to_ix[w] for w in context]
    context_var = autograd.Variable(torch.LongTensor(context_idxs)).cuda()
    log_probs, embedd = model(context_var)
    tagert = autograd.Variable(torch.LongTensor([word_to_ix[target]])).cuda()
    loss = loss_function(log_probs, tagert)
    total_loss += loss.data
losses.append(total_loss)
b = open('test.txt', 'w', encoding='utf-8')
b.write(str(ix_to_word) + '\n' + str(word_to_ix) + '\n')
for i in range(0, len(vocab)):
    lookup_tensor = torch.LongTensor([i]).cuda()
    b.write(ix_to_word[i] + str(embedd(autograd.Variable(lookup_tensor))) + '\n')
    # print(ix_to_word[i])
    # print(embedd(autograd.Variable(lookup_tensor)))
b.close()
