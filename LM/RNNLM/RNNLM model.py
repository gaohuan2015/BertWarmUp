import torch
import torch.nn as nn
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_
from RNNLM.util.data import Corpus
import RNNLM.util.const as con
import RNNLM.util.model as RNNLM

device = torch.device('cpu')
corpus = Corpus()
ids = corpus.get_data('data.txt', con.batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // con.seq_length

model = RNNLM(vocab_size, con.embed_size, con.hidden_size, con.num_layers)
criterion = nn.CrossEntropyLoss() #交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=con.learning_rate)
# 反向传播过程“截断”(不复制gradient)
def detach(states):
    return [state.detach() for state in states]

# 训练模型
for epoch in range(con.num_epochs):
    states = (torch.zeros(con.num_layers, con.batch_size, con.hidden_size),
              torch.zeros(con.num_layers, con.batch_size, con.hidden_size))
#???states
    for i in range(0, ids.size(1) - con.seq_length, con.seq_length):
        inputs = ids[:, i:i + con.seq_length]
        targets = ids[:, (i + 1):(i + 1) + con.seq_length]

        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i + 1) // con.seq_length
        if step % 100 == 0:
            print('Epoch {}, Loss: {:.4f}, Perplexity: {:.2f}'
                  .format(epoch + 1, loss.item(), np.exp(loss.item())))

with torch.no_grad():
    with open('sample.txt', 'w') as f:
        state = (torch.zeros(con.num_layers, 1, con.hidden_size),
                 torch.zeros(con.num_layers, 1, con.hidden_size))

        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1)

        for i in range(con.num_samples):
            output, state = model(input, state)
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            input.fill_(word_id)
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i + 1, con.num_samples, 'sample.txt'))

#torch.save(model.state_dict(), 'model.ckpt')