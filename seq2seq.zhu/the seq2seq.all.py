# seq2seq with attention for translation
# In this project: a neural network to translate from French to English.
from __future__ import unicode_literals, print_function, division
import random
import matplotlib.pyplot as plt
from util import dataset as ds, Net
from util.seq2seqmodel import seq2seqmodel


input_lang, output_lang, pairs = ds.prepareData('eng', 'fra', True)
print(random.choice(pairs))
hidden_size = 256
encoder1 = Net.EncoderRNN(input_lang.n_words, hidden_size)
attn_decoder1 = Net.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
seq2seqmodel = seq2seqmodel(input_lang,output_lang,pairs,encoder1,attn_decoder1)

seq2seqmodel.trainIters(10000, print_every=500)

output_words, attentions = seq2seqmodel.evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())
