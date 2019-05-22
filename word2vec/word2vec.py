import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.autograd import Variable




class CBOW(nn.Module):
    def __init__(self, juzi_size, yujing_size, qianru_size, hidden_size):
        super(CBOW, self).__init__()
        self.embed_layer = nn.Embedding(juzi_size, qianru_size)
        self.linear_1 = nn.Linear(2 * yujing_size * qianru_size, hidden_size)
        self.linear_1_5 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, juzi_size)

    def forward(self, input_data):
        embeds = self.embed_layer(input_data).view((1, -1))
        output = F.relu(self.linear_1(embeds))
        output = F.relu(self.linear_1_5(output))
        output = F.log_softmax(self.linear_2(output))
        return output


yujing = 2
qianru = 32
hidden = 128
jindu = 0.0001
step = 15




def context_to_tensor(context, idx_dict):
    """ Converts context list to tensor. """
    context_idx = [idx_dict[word] for word in context]

    return Variable(torch.LongTensor(context_idx))


model1 = 0
if model1 == 0:
    word = open('leta.en', 'r', encoding='utf-8')
    word2 = open('leta2.en', 'r', encoding='utf-8')
    word = word.read().split()
    word2 = word2.read().split()
    word1 = set(word)
    word3 = set(word2)
    word_to_idx = {word: idx for idx, word in enumerate(word3)}
    print(word_to_idx)
    data = list()
    model = CBOW(len(word3), yujing, qianru, hidden)
    optimizer = opt.Adam(model.parameters(), lr=jindu)
    loss_function = nn.NLLLoss()
    if torch.cuda.is_available():
        MyModel = model.cuda()
    for i in range(2, len(word1) - 2):
        # Context, target
        bow = ([word[i - 2], word[i - 1], word[i + 1], word[i + 2]], word[i])
        data.append(bow)
    print(data)
    for e in range(step):
        total_loss = torch.FloatTensor([0])
        relly = 0
        i = 0
        for bag in data:
            i += 1
            # Get data and labels
            context_data = context_to_tensor(bag[0], word_to_idx).cuda()
            target_data = Variable(torch.LongTensor([word_to_idx[bag[1]]])).cuda()
            # Do forward pass
            model.zero_grad()
            prediction = model(context_data)
            loss = loss_function(prediction, target_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            # print(total_loss)
            if (prediction[0, word_to_idx[bag[1]]] == torch.max(prediction)):
                relly += 1
            else:
                relly += 0
        # Bookkeeping

    print(relly / i)
    print('Step: {} | Loss: {}'.format(e, total_loss.numpy()))
    torch.save(model.state_dict, "word2vec.pth")

    word = open('leta1.en', 'r', encoding='utf-8')
    word = word.read().split()
    word1 = set(word)
    # model2 = CBOW(len(word1), yujing, qianru, hidden)
    # model2.load_state_dict(torch.load('word2vec.pth'))
    # word_to_idx = {word: idx for idx, word in enumerate(word1)}
    data = list()
    for i in range(2, len(word1) - 2):
        # Context, target
        bow = ([word[i - 2], word[i - 1], word[i + 1], word[i + 2]], word[i])
        data.append(bow)
    print(data)
    total_loss=0
    loss=0
    for bag in data:
        context_data = context_to_tensor(bag[0], word_to_idx).cuda()
        target_data = Variable(torch.LongTensor([word_to_idx[bag[1]]])).cuda()
        prediction = model(context_data)
        loss = loss_function(prediction, target_data)
        total_loss += loss.data
        # print(bag)
        # print(bag[1])
        # print(prediction[0, word_to_idx[bag[1]]] )
        # print(torch.max(prediction))
        # print('')

    print('Loss: {}'.format( total_loss))
