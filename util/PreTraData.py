# Training
# Preparing Training Data
from __future__ import unicode_literals, print_function, division
import torch
from torch.autograd import Variable
from util import ConfigOp as cop


class PreTraData:
    def __init__(self,input_lang,output_lang):
        self.input_lang=input_lang
        self.output_lang = output_lang

    def indexesFromSentence(self,lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def variableFromSentence(self,lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(cop.EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        return result

    def variablesFromPair(self,pair):
        input_variable =  self.variableFromSentence(self.input_lang, pair[0])
        target_variable =  self.variableFromSentence(self.output_lang, pair[1])
        return (input_variable, target_variable)

    # Training the Model





    



