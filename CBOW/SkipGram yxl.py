import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


CONTEXT_SIZE = 3
TARGET_SIZE=1
#raw_text = open('leta.en',encoding='utf-8').\
#        read().strip().split()
raw_text = """Motorists have also been asked to avoid the area around the complex where more than 39,000 employees work - about one third of BASF's total global workforce.
Scientists' manifesto hoped to help government realize potential of innovations for economic development RIGA, Oct 17 (LETA) - Scientists and academicians are planning to submit to Prime Minister Maris Kucinskis (Greens/Farmers) a manifesto explaining that scientists, in close cooperation with businessmen, can create innovations contributing to the economic development of Latvia.
Scientists hope that the manifesto will help the government realize how scientific innovations reach businessmen and contribute to the economic growth of the country, Latvian Academy of Sciences President Ojars Sparitis emphasized in an interview with the LNT television this morning.
At the moment, few politicians understand that scientists can help businessmen create new products and generate profits in the future, said Sparitis.
In addition, none of the coalition parties have engaged in a meaningful dialog with scientists, while the Education and Science Ministry and Ministry of Economics have not been doing enough, added Sparitis.
Leonardo DiCaprio Urged to Step Down From UN Climate Change Role The Hollywood reporter A rainforest charity calls on the star to denounce his connection to individuals involved in a Malaysian corruption scandal and return laundered money he allegedly received or give up his role.
"If DiCaprio is unwilling to come clean, we ask him to step down as UN Messenger for Peace for climate change, because he simply lacks the credibility for such an important role," said Lukas Straumann, director of the Switzerland-based charity, which has a particular focus on deforestation in Malaysia.
Leonardo DiCaprio, the Malaysian Money Scandal and His "Unusual" Foundation At the press conference, entitled "Recovery of Stolen Malaysian Assets," a direct link was made between the 1MDB corruption scandal and major environmental issues in Malaysia, such as deforestation, one of the main concerns of the Leonardo DiCaprio Foundation.
DiCaprio had been invited to talk at the press conference via an open letter from the Bruno Manser Funds, but did n't respond.
"We ca n't save the environment if we fail to stop corruption," said Straumann, who called DiCaprio's criticism of deforestation in Indonesia, just across a piece of water called the Malacca Strait from Malaysia, "cynical hypocrisy ."
Bosnian Serb elected Srebrenica mayor: official results Sarajevo, Oct 17, 2016 (AFP) - A Bosnian Serb has been elected mayor of Srebrenica, where thousands of Muslims were killed by the Bosnian Serb army during the 1990s war, official results showed Monday.
Mladen Grujicic will be the first Serbian mayor of the eastern Bosnian town since 1999 after winning 54.4 percent of votes in local elections on October 2.
The July 1995 massacre of some 8,000 Muslim men and boys from Srebrenica, a UN-protected enclave at the time, was Europe's worst atrocity since World War II and determined by two international courts to be genocide.
Unlike many European countries where abdication of kings and queens are relatively common, Japan's modern imperial law does n't allow abdication, and Japan's postwar Constitution stipulates the emperor as mere "symbol" with no political power or a say.
Allowing Akihito to abdicate would be a major change to the system, and raises a series of legal and logistical questions, ranging from laws subject to change to the emperor's post-abdication role, his title and residence.
His message was subtle and the Emperor did not use the word "abdication," because saying that openly could have violated his Constitutional status.
The government reportedly wants to allow Akihito's abdication as an exception and enact a special law to avoid dealing with divisive issues such as possible female succession and lack of successors.
The paper cited unidentified Japanese and Russian government sources.
"We deny the Nikkei report that Japan and Russia are discussing the joint administration of the Northern Territories," Japanese foreign ministry spokesman Yasuhira Kawamura told Reuters in an email, referring to the islands off Hokkaido known in Russia as the Southern Kuriles.
A Japanese ruling party source told Reuters that a proposal such as that reported by Nikkei was not very realistic.
The newspaper said Tokyo hoped to negotiate a return of the two smaller islands while adopting joint control of two larger islands.
Japan's position has been to assert its sovereignty over all of the disputed islands, but Russia regularly cites a 1956 joint declaration, never put into effect, that stipulates the return of the two smaller isles to Japan.
Elering, Baltic Connector OY sign agreement to build gas interconnection TALLINN, Oct 17, BNS – The Estonian state owned transmission system operator Elering AS and the Finnish state owned company Baltic Connector OY on Monday signed an agreement on their cooperation concerning the undersea portion of the Balticconnector project to interconnect the natural gas transmission systems of Estonia and Finland.
The parties will prepare a joint procurement strategy for constructing the offshore section; the on-shore sections of the pipeline will be developed separately by each party.
Elering and Baltic Connector will establish a joint steering group, which serves as the highest decision-making body in questions regarding the common issues related to the project.
Phil Collins announces European comeback tour London, Oct 17, 2016 (AFP) - Veteran British singer Phil Collins announced on Monday he is coming out of retirement with a comeback tour next summer, despite battling with injury and alcoholism.
"I stopped work because I wanted to be a dad at home.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)

word_to_ix = {word: i for i, word in enumerate(vocab)}
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
data = []
for i in range(2, len(raw_text) - 2):
    target = raw_text[i]
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    data.append((target,context))
print(data[:5])


class SkipGram(nn.Module):
    def __init__(self, n_word, n_dim,target_size):
        super(SkipGram, self).__init__()
        #实例化
        self.embeddings = nn.Embedding(n_word, n_dim)  #embedding把输入的词变为向量
        self.linear1 = nn.Linear(50, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((4, -1))#view(1,-1)是把字典转化到一行
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


model = SkipGram(len(word_to_ix), 200, TARGET_SIZE)
losses = []
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    print('epoch{}'.format(epoch))
    print('*'*10)
    total_loss = 0
    for target,context, in data:
        target = Variable(torch.LongTensor([word_to_ix[target]]))
        context = Variable(torch.LongTensor([word_to_ix[i] for i in context]))
        out = model(target)
        loss = loss_function(out, context)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('loss: {:.6f}'.format(total_loss / len(data)))
