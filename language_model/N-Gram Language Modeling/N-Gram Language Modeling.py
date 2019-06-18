import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence =  open('data.txt', encoding='utf-8').\
    read().split()

trigrams = [([test_sentence[i],test_sentence[i+1]],test_sentence[i+2])
            for i in range(len(test_sentence)-2)]

print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)} 

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

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(),lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype = torch.long)
        target_idxs = torch.tensor([word_to_ix[target]],dtype=torch.long)
        model.zero_grad()

        log_probs = model(context_idxs)
        loss = loss_function(log_probs,target_idxs)
        loss.backward()
        optimizer.step()
       
        total_loss += loss.item()
    losses.append(total_loss)
print('losses:', losses) 
