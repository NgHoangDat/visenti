import sys
sys.path.append("..")
import argparse

from six.moves import cPickle

import viprocessor as vp
from train import train_model


def main(
    data_files, we_folder, saveto, max_epochs=100,
    dim=300, cv=10, limit=None, filter_hs=[3, 4, 5], we='randw2vglove',
    valid_portion=0.1, decay_c=0, encoder='cnnlstm', batchsize=25,
    deep=0, rnnlayer=1, feature_maps=100, rnnshare=False
):
    pad = max(filter_hs) - 1
    cPickle.dump(vp.build_dataset(
        data_folder=data_files,
        we_folder=we_folder,
        dim=dim,
        cv=cv,
        limit=limit
    ), open(saveto, "wb"))    
    revs, word_embeding, word_idx_map, vocab, max_l = cPickle.load(open(saveto, "rb"))

    Ws = []
    if 'w2v' in we:
        print('loading word2vec...')
        Ws.append(word_embeding['w2v'])

    if 'rand' in we:
        print('loading random...')
        Ws.append(word_embeding['rand'])

    if 'glove' in we:
        print('loading glove...')
        Ws.append(word_embeding['glove'])

    vocab_size, dim = Ws[0].shape
    vocab_size -=1

    train_model(
        revs=revs,
        word_idx_map=word_idx_map,
        max_l=max_l,
        pad=pad,
        valid_portion = valid_portion,
        max_epochs=max_epochs,
        dim_proj = dim,
        decay_c = decay_c,
        n_words = vocab_size,
        W = Ws,
        encoder = encoder,
        batch_size = batch_size,
        deep = deep,
        rnnlayer = rnnlayer,
        filter_hs = filter_hs,
        feature_maps = feature_maps,
        rnnshare = rnnshare
    )



def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-datapath', default='../data/data.p')
    parser.add_argument('-pos', type=str, default="../data/en-pos")
    parser.add_argument('-neg', type=str, default="../data/en-neg")
    parser.add_argument('-wef', type=str,  default="../data/")
    parser.add_argument('-dim', type=int,  default=300)
    parser.add_argument('-cv', type=int, default=10)
    parser.add_argument('-limit', type=int, default=-1)
    parser.add_argument('-We',    default='w2vgloverand', help='Word Embedding')
    parser.add_argument('-bidir',   type=bool, default=False, help='Bidirectional RNN')
    parser.add_argument('-rnnshare',type=bool, default=False, help='')
    parser.add_argument('-rnnlayer',type=int, default=1)
    parser.add_argument('-deep',    type=int, default=0)
    parser.add_argument('-encoder', default='cnnlstm')
    parser.add_argument('-optim',   default='adadelta')
    parser.add_argument('-dropout_penul', type=float, default=0.5)
    parser.add_argument('-decay_c', type=float, default=0)
    parser.add_argument('-pool_type', default='max')
    parser.add_argument('-filter_hs', default='345')
    parser.add_argument('-combine',type=bool, default=False, help='')
    parser.add_argument('-feature_maps', type=int, default=100)
    parser.add_argument('-rm_unk', type=bool, default=False, help='')
    parser.add_argument('-validportion', type=float, default=0.15)
    parser.add_argument('-batchsize', type=int, default=25)
    parser.add_argument('-init',  default='uniform')
    parser.add_argument('-salstm', type=bool, default=False, help='')
    parser.add_argument('-max_epochs', type=int, default=10)
    args = vars(parser.parse_args())

    filter_hs = [int(h) for h in list(args['filter_hs'])]
    pad = max(filter_hs) - 1
    cPickle.dump(vp.build_dataset(
        data_folder=[
            args['neg'],
            args['pos']
        ],
        we_folder=args['wef'],
        dim=args['dim'],
        cv=args['cv'],
        limit=args['limit']
    ), open(args['datapath'], "wb"))    
    revs, word_embeding, word_idx_map, vocab, max_l = cPickle.load(open(args['datapath'],"rb"))

    Ws = []
    if 'w2v' in args['We']:
        print('loading word2vec...')
        Ws.append(word_embeding['w2v'])

    if 'rand' in args['We']:
        print('loading random...')
        Ws.append(word_embeding['rand'])

    if 'glove' in args['We']:
        print('loading glove...')
        Ws.append(word_embeding['glove'])

    vocab_size, dim = Ws[0].shape
    vocab_size -=1

    train_model(
        revs=revs,
        word_idx_map=word_idx_map,
        max_l=max_l,
        pad=pad,
        valid_portion = args['validportion'],
        max_epochs=100,
        dim_proj = dim,
        decay_c = args['decay_c'],
        n_words = vocab_size,
        W = Ws,
        encoder = args['encoder'],
        batch_size = args['batchsize'],
        deep = args['deep'],
        rnnlayer = args['rnnlayer'],
        filter_hs = filter_hs,
        dropout_penul = args['dropout_penul'],
        pool_type = args['pool_type'],
        combine = args['combine'],
        feature_maps = args['feature_maps'],
        init = args['init'],
        optimizer = args['optim'],
        rnnshare = args['rnnshare'],
        bidir = args['bidir'],
        salstm = args['salstm'],
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-datapath', default='../data/data.p')
    parser.add_argument('-pos', type=str, default="../data/en-pos")
    parser.add_argument('-neg', type=str, default="../data/en-neg")
    parser.add_argument('-wef', type=str,  default="../data/")
    parser.add_argument('-dim', type=int,  default=300)
    parser.add_argument('-cv', type=int, default=10)
    parser.add_argument('-limit', type=int, default=-1)
    parser.add_argument('-We',    default='w2vgloverand', help='Word Embedding')
    parser.add_argument('-rnnshare',type=bool, default=False, help='')
    parser.add_argument('-rnnlayer',type=int, default=1)
    parser.add_argument('-deep',    type=int, default=0)
    parser.add_argument('-encoder', default='cnnlstm')
    parser.add_argument('-decay_c', type=float, default=0)
    parser.add_argument('-filter_hs', default='345')
    parser.add_argument('-feature_maps', type=int, default=100)
    parser.add_argument('-validportion', type=float, default=0.15)
    parser.add_argument('-batchsize', type=int, default=25)
    parser.add_argument('-max_epochs', type=int, default=10)
    args = vars(parser.parse_args())
    data_files = [
        args['neg'],
        args['pos']
    ]
    main(
        data_files=data_files,
        we_folder=args['wef'],
        saveto=args['datapath']
    )
