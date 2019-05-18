import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

CONTEXT_SIZE = 2
raw_text = """The document includes a payload that downloads malware, which is designed to target online banking information.
Russia and Belarus to conduct observation flights over Latvia RIGA, Sept 16 (LETA) - From September 19 to 21, in accordance with the Open Skies Agreement, Russian and Belarussian military representatives, together with representatives from the Latvian armed forces will conduct observation flights over Latvia.
Afterwards, the Russian and Belarussian military representatives will go to Lithuanian to conduct observation flights in this country, LETA was informed by the Defense Ministry's press department.
The Treaty on Open Skies is part of security and confidence building measures among member states of the Organization for Security and Cooperation in Europe (OSCE).
Housing and employment were found for all asylum seekers - Red Cross RIGA, Sept 16 (LETA) - Mentors from the Latvian Red Cross had found housing and employment for all of the asylum seekers that they had worked with, Latvian Red Cross spokesman Uldis Likops told LETA.
The apartments were without amenities and the jobs were not the best paying ones - which is why the asylum seekers decided to refuse the offers made to them and leave the country.
For example, the plan does not even foresee that asylum seekers could have their own point of view and agree not agree to something.
''It is possible that these asylum seekers were not been informed from the very start that we can offer then services available only in Latvia, apartment for Latvian prices and jobs that are not well paid.
Despite difficulties, Red Cross mentors had found potential housing and employment for all of the asylum seekers they were working with - five families totaling 23 persons.
''The asylum seekers decided they will not do these jobs.
Telia Company has suggested to merge the assets and build a New Generation Telco also in Latvia as Telia Company has done in other markets in Nordics and Baltics.
"We want to change the ownership structure or the governance of both companies, Lattelecom and LMT to fulfill Telia's strategy and run businesses as normal in Latvia.
He has worked 18 years for Telia Company in Finland, Eurasia and Lithuania where Telia Company has successfully joint forces of the local companies Omnitel and TEO.
""".split()

# lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
       # read().strip().split('\n')

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)

word_to_ix = {word:i for i, word in enumerate(vocab)}
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
data = []
for i in range(2,len(raw_text)-2):
    context = [raw_text[i-2],raw_text[i-1],raw_text[i+1],raw_text[i+2]]
    target = raw_text[i]
    data.append((context,target))
print(data[:5])

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW,self).__init__()
        self.embeddings = nn.Embedding(n_word,n_dim)
        self.linear1 = nn.Linear(2*context_size*n_dim,128)
        self.linear2 = nn.Linear(128,n_word)
    def forward(self,inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out,dim=1)
        return log_probs

model = CBOW(len(word_to_ix),100,CONTEXT_SIZE)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

for epoch in range(100):
    print('epoch{}'.format(epoch))
    print('*'*10)
    total_loss = 0
    for context, target in data:
        context = Variable(torch.LongTensor([word_to_ix[i] for i in context]))
        target = Variable(torch.LongTensor([word_to_ix[target]]))
        out = model(context)
        loss = loss_function(out,target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss: {:.6f}'.format(total_loss / len(data)))
