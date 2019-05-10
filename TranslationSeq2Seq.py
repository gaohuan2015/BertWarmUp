# date: 2018/4/21
# author: luochaoyang
# seq2seq with attention for translation
# In this project: a neural network to translate from French to English.
from __future__ import unicode_literals, print_function, division

import random
import matplotlib.pyplot as plt
from util import DataSet as ds, net
from util.Seq2Seql import Seq2Seq

#the Seq2SeqModel parameter
hidden_size = 256
learning_rate=0.001
#the predata Oparation
input_lang, output_lang, pairs = ds.prepareData('eng', 'fra', True)
print(random.choice(pairs))

#new EncoderRNN and AttnDecoderRNN
encoder1 = net.EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = net.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)


Seq2Seq = Seq2Seq(input_lang,output_lang,pairs,encoder1,attn_decoder1,learning_rate=learning_rate)


#train
Seq2Seq.seq2seqmodel.load()
Seq2Seq.trainIters(1, print_every=500)

#evaluate
output_words, attentions = Seq2Seq.evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())

Seq2Seq.evaluateRandomly(encoder1, attn_decoder1)

Seq2Seq.evaluateAndShowAttention("elle a cinq ans de moins que moi .")

Seq2Seq.evaluateAndShowAttention("elle est trop petit .")

Seq2Seq.evaluateAndShowAttention("je ne crains pas de mourir .")

Seq2Seq.evaluateAndShowAttention("c est un jeune directeur plein de talent .")