import torch
import os
#字典
class Dictionary(object):

    def __init__(self):
        # 构建word2id,id2word两个字典
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

#语料库
class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary() #创建字典类对象

    def get_data(self, path, batch_size=20):
        with open(path, 'r',encoding='utf8') as f:
            tokens = 0
            for line in f:  #遍历文件中的每一行
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)


        ids = torch.LongTensor(tokens)# 对文件做Tokenize
        token = 0
        with open(path, 'r',encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        return ids.view(batch_size, -1)

