from __future__ import unicode_literals, print_function, division
import random
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
import Train as T
import DataSet as SD
import seq2seqModel as sm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, max_length=SD.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = T.tensorFromSentence(SD.input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SD.SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == SD.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(SD.output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(SD.pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
hidden_size = 256
encoder1 = sm.EncoderRNN(SD.input_lang.n_words, hidden_size).to(device)
attn_decoder1 = sm.AttnDecoderRNN(hidden_size, SD.output_lang.n_words, dropout_p=0.1).to(device)

T.trainIters(encoder1, attn_decoder1, 1000, print_every=500)