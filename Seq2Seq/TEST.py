from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import random
import jieba
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.modules as mod
from torch import optim
import torch.nn.functional as F
SOS_token = 0
EOS_token = 1
eng_prefixes = (
    "i am", "i m",
    "he is", "he s",
    "she is", "she s",
    "you are", "you re",
    "we are", "we re",
    "they are", "they re"
    # "as","the","we","that"
)
hidden_size=512
MAX_LENGTH =20
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat([embedded[0], hidden[0]], 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat([embedded[0], attn_applied[0]], 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
def unicode2Ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicode2Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # print(s)
    return s

def readLangs(lang1, lang2,dz, reverse=False):
    print("Reading lines...")
    dz = dz
    # Read the file and split into lines
    lines = open(dz, encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # print(pairs)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs



def filterPair(p):
    return len(p[0].split(' '))< MAX_LENGTH and  len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2,dz, reverse=False):
    dz=dz
    input_lang, output_lang, pairs = readLangs(lang1, lang2,dz, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1,1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_word_vector(s1,s2):
    """
    :param s1: 句子1
    :param s2: 句子2
    :return: 返回句子的余弦相似度
    """
    # 分词
    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)
    list_word1 = (','.join(cut1)).split(',')
    list_word2 = (','.join(cut2)).split(',')

    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))

    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 输出向量
    # print(word_vector1)
    # print(word_vector2)
    return word_vector1, word_vector2

def cos_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1=float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist1

def filter_html(html):
    """
    :param html: html
    :return: 返回去掉html的纯净文本
    """
    dr = re.compile(r'<[^>]+>',re.S)
    dd = dr.sub('',html).strip()
    return dd

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    loss=0
    output=0
    for i in range(n):
        pair = random.choice(pairs)
        # pair=i
        # print('>', pair[0])
        # print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        # print('<', output_sentence)
        # print(i)
        s1 = pair[1]
        s2 = output_sentence
        vec1, vec2 = get_word_vector(s1, s2)
        dist1 = cos_dist(vec1, vec2)
        loss+=dist1
    print(loss/len(pairs))
def evaluate1(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly1(encoder, decoder, n=10):
    output=0
    loss=0
    for i in range(n):
        pair = random.choice(pairs)
        # pair=i
        # print('>', pair[0])
        # print('=', pair[1])
        output_words = evaluate1(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        # print('<', output_sentence)
        # print('')
        s1 = pair[1]
        s2 = output_sentence
        vec1, vec2 = get_word_vector(s1, s2)
        dist1 = cos_dist(vec1, vec2)
        loss += dist1
    print(loss / len(pairs))


model=1
if model==0:
    # print(random.choice(pairs))
    T1 = torch.load('EncoderRNN.pt', map_location='cpu')
    T1.eval()
    # EncoderRNN.load_state_dict(torch.load('EncoderRNN.pt'))
    # EncoderRNN.eval()
    # DecoderRNN = torch.load('DecoderRNN.pt')
    # DecoderRNN.load_state_dict(torch.load( 'DecoderRNN.pt'))
    # DecoderRNN.eval()
    T3= torch.load('AttnDecoderRNN.pt', map_location='cpu')
    T3.eval()
    # AttnDecoderRNN.load_state_dict(torch.load('AttnDecoderRNN.pt'))
    # AttnDecoderRNN.eval()
    input_lang, output_lang, pairs = prepareData('eng', 'fra', dz='LETA-lv-en/eng-fra.txt', reverse=True)
    print(len(pairs))
    print(type(input_lang))
    # print(len(input_lang))
    # c=torch.tensor(input_lang.n_words)

    # encoder1 =T1(c, hidden_size)
    # attn_decoder1 = T3(hidden_size, output_lang.n_words, dropout_p=0.1)
    evaluateRandomly(T1, T3, n=len(pairs))
elif model==1:
    # print(random.choice(pairs))
    T1 = torch.load('EncoderRNN01.pt', map_location='cpu')
    T1.eval()
    # EncoderRNN.load_state_dict(torch.load('EncoderRNN.pt'))
    # EncoderRNN.eval()
    # DecoderRNN = torch.load('DecoderRNN.pt')
    # DecoderRNN.load_state_dict(torch.load( 'DecoderRNN.pt'))
    # DecoderRNN.eval()
    T3 = torch.load('DecoderRNN01.pt', map_location='cpu')
    T3.eval()
    # AttnDecoderRNN.load_state_dict(torch.load('AttnDecoderRNN.pt'))
    # AttnDecoderRNN.eval()
    input_lang, output_lang, pairs = prepareData('eng', 'fra', dz='LETA-lv-en/eng-fra.txt', reverse=True)
    print(len(pairs))
    print(type(input_lang))
    # print(len(input_lang))
    # c=torch.tensor(input_lang.n_words)

    # encoder1 =T1(c, hidden_size)
    # attn_decoder1 = T3(hidden_size, output_lang.n_words, dropout_p=0.1)
    evaluateRandomly1(T1, T3, n=len(pairs))

