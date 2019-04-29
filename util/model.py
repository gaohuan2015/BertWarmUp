from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from util import ConfigOp as cop
from torch import optim

class BaseModel(nn.Module):
    def __init__(self, name, path='.\ModelData'):
        super(BaseModel, self).__init__()

        self.name = name
        self.iteration = 0
        self.seq2seq_Ecoder_path = os.path.join(path, (name + '_Ecoder_path'))
        self.seq2seq_Decoder_path = os.path.join(path, (name + '_Decoder_path'))

    def load(self):
        if os.path.exists(self.seq2seq_Ecoder_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.seq2seq_Ecoder_path)
            else:
                data = torch.load(self.seq2seq_Ecoder_path, map_location=lambda storage, loc: storage)
            self.encoder1.load_state_dict(data['encoder1'])
        if os.path.exists(self.seq2seq_Decoder_path):
            if torch.cuda.is_available():
                data = torch.load(self.seq2seq_Decoder_path)
            else:
                data = torch.load(self.seq2seq_Decoder_path, map_location=lambda storage, loc: storage)
            self.attn_decoder1.load_state_dict(data['attn_decoder1'])

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'encoder1': self.encoder1.state_dict()
        }, self.seq2seq_Ecoder_path)

        torch.save({
            'attn_decoder1': self.attn_decoder1.state_dict()
        }, self.seq2seq_Decoder_path)



class Seq2SeqModel(BaseModel):

    def __init__(self,encoder1,attn_decoder1,teacher_forcing_ratio,learning_rate,name='Seq2SeqModel'):
        super(Seq2SeqModel, self).__init__(name)
        self.teacher_forcing_ratio=teacher_forcing_ratio
        self.encoder1=encoder1
        self.attn_decoder1=attn_decoder1
        self.encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=learning_rate)

    def mytrain(self,input_variable, target_variable,criterion,
              max_length=cop.MAX_LENGTH):
        encoder=self.encoder1
        decoder=self.attn_decoder1
        self.iteration+=1
        encoder_hidden = encoder.initHidden()
        encoder_optimizer=self.encoder_optimizer
        decoder_optimizer=self.decoder_optimizer
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        #  encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[cop.SOS_token]]))
        #  decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                # decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
                if ni == cop.EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data / target_length









