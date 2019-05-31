import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

#text_sentence = open('leta.en',encoding='utf-8').\
#        read().split()
test_sentence = """Motorists have also been asked to avoid the area around the complex where more than 39,000 employees work - about one third of BASF's total global workforce.
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

# we should tokenize(固化) the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)

trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
print(trigrams[:3])# print the first 3, just so you can see what they look like

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long) # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words into integer indices and wrap them in tensors)
        model.zero_grad() # Recall that torch *accumulates* gradients. Before passing in a new instance, you need to zero out the gradients from the old
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()# Do the backward pass and update the gradient
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)