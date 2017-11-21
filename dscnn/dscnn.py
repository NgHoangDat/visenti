from six.moves import cPickle as pkl
import sys
import argparse
from train import train_model


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    def str2bool(v):
        return v.lower() in ('true','1')
    parser.register('type','bool',str2bool)

    parser.add_argument('-datapath', default='../data/data.p')
    parser.add_argument('-We',    default='w2vgloverand', help='Word Embedding')
    parser.add_argument('-bidir',   type='bool', default=False, help='Bidirectional RNN')
    parser.add_argument('-rnnshare',type='bool', default=False, help='')
    parser.add_argument('-rnnlayer',type=int, default=1)
    parser.add_argument('-deep',    type=int, default=0)
    parser.add_argument('-encoder', default='cnnlstm')
    parser.add_argument('-optim',   default='adadelta')
    parser.add_argument('-dropout_penul', type=float, default=0.5)
    parser.add_argument('-decay_c', type=float, default=0)
    parser.add_argument('-pool_type', default='max')
    parser.add_argument('-filter_hs', default='345')
    parser.add_argument('-combine',type='bool', default=False, help='')
    parser.add_argument('-feature_maps', type=int, default=100)
    parser.add_argument('-rm_unk',type='bool', default=False, help='')
    parser.add_argument('-validportion', type=float, default=0.15)
    parser.add_argument('-batchsize', type=int, default=100)
    parser.add_argument('-init',  default='uniform')
    parser.add_argument('-salstm',type='bool', default=False, help='')
    parser.add_argument('-max_epochs', type=int, default=10)
    args = vars(parser.parse_args())
    #print(args)
    
    filter_hs = [int(h) for h in list(args['filter_hs'])]
    pad = max(filter_hs) - 1

    x = pkl.load(open(args['datapath'],"rb"))
    revs, word_embeding, word_idx_map, vocab, max_l = x[0], x[1], x[2], x[3], x[4]

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
	
    perf = train_model(
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
    p = "%.2f\t%.2f\t%.2f\n" % perf

    print(p)
