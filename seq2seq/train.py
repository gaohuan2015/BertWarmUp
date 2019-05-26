import os
import math
import argparse
import torch
import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
# from model_atten import Encoder, Decoder, Seq2Seq
from model_sim import Encoder, Decoder, Seq2Seq
from utils import load_dataset


def cos_distance(x, y):
    vec_inner_product = np.dot(x, y)
    x_morm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    cos_dis = vec_inner_product / (x_morm * y_norm)
    return cos_dis


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    '''
    add_argument()方法，用来指定程序需要接受的命令参数
    add_argument()常用的参数：
    dest：如果提供dest，例如dest="a"，那么可以通过args.a访问该参数
    default：设置参数的默认值
    action：参数出发的动作
    store：保存参数，默认
    store_const：保存一个被定义为参数规格一部分的值（常量），而不是一个来自参数解析而来的值。
    store_ture/store_false：保存相应的布尔值
    append：将值保存在一个列表中。
    append_const：将一个定义在参数规格中的值（常量）保存在一个列表中。
    count：参数出现的次数
    version：打印程序版本信息
    type：把从命令行输入的结果转成设置的类型
    choice：允许的参数值
    help：参数命令的介绍
    '''
    p.add_argument('-epochs', type=int, default=1, help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32, help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0, help='in case of gradient explosion')
    return p.parse_args()


def evaluate(model, val_iter, DE, EN):
    np_trg = []
    np_tra = []
    simility_all = 0
    model.eval()
    translates = []
    pad = EN.vocab.stoi['<pad>']
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        # src = Variable(src.data.cuda(), volatile=True)
        # trg = Variable(trg.data.cuda(), volatile=True)
        src = Variable(src.data, volatile=True)
        trg = Variable(trg.data, volatile=True)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        _, predicted = torch.max(output, 2)   # (length, batchsize, 8011)
        for i in range(len(len_src)):
            translate = ''
            for x in predicted[:,i]:
                translate += DE.vocab.itos[x] + ' '
            translates.append(translate)
            np_tra.append(predicted[:,i].numpy())
        for i in range(len(len_trg)):
            np_trg.append(trg.data[:, i].numpy())
    for i in range(len(val_iter)):
        simility = cos_distance(np_tra[i], np_trg[i])
        simility_all += simility
    return simility_all / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, DE, EN):
    model.train()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        # src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(output[1:].view(-1, vocab_size), trg[1:].contiguous().view(-1), ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data
        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f]\t[pp:%5.2f]" % (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    # assert torch.cuda.is_available()
    print("******************** preparing dataset **********************")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[TRAIN]: %d (dataset:%d)\t[TEST]: %d (dataset:%d)" % (len(train_iter), len(train_iter.dataset), len(test_iter), len(test_iter.dataset)))
    print("[DE_vocab]: %d\t\t\t\t[En_vocab]: %d" % (de_size, en_size))
    print("******************** Instantiating models ********************")

    encoder = Encoder(de_size, embed_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size, n_layers=1, dropout=0.5)
    # seq2seq = Seq2Seq(encoder, decoder).cuda()
    seq2seq = Seq2Seq(encoder, decoder)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)
    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter, en_size, args.grad_clip, DE, EN)
        val_loss = evaluate(seq2seq, val_iter, DE, EN)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS" % (e, val_loss, math.exp(val_loss)))
        if not best_val_loss or val_loss < best_val_loss:
            print("******************** saving model ********************")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, DE, EN)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
