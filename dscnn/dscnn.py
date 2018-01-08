import sys
sys.path.append("..")
# import argparse

from six.moves import cPickle
from train import train_model


def main(
    data_file, report_to,
    max_epochs=100, filter_hs=[3, 5, 7], we='randw2vglove',
    valid_fold=0, batch_size=25, feature_maps=10
):
    pad = max(filter_hs) - 1
    
    revs, word_embeding, word_idx_map, vocab, max_l = cPickle.load(open(data_file, "rb"))

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
        report_to=report_to,
        valid_fold=valid_fold,
        max_epochs=max_epochs,
        dim_proj=dim,
        n_words=vocab_size,
        W=Ws,
        batch_size=batch_size,
        filter_hs=filter_hs,
        feature_maps=feature_maps,
    )


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-report', type=str,  default="../data/foody/report_dscnn")
    # parser.add_argument('-filter_hs', default='357')
    # parser.add_argument('-feature_maps', type=int, default=10)
    # parser.add_argument('-valid_fold', type=int, default=0)
    # parser.add_argument('-batch_size', type=int, default=25)
    # parser.add_argument('-max_epochs', type=int, default=25)
    # args = vars(parser.parse_args())
    

    # for dim in (10, 25, 50, 100, 150, 200, 250, 300):
    # data_file = '../data/foody/data' + str(dim) + '.p' 
    #dim = 100
    # options = [
    #    [3, 3, 3],
    #    [5, 5, 5],
    #    [7, 7, 7],
    #    [3, 4, 5],
    #    [3, 5, 7]
    #]
    # i = sys.argv[1]
    dim = 100
    print("Start")
    main(
        data_file='../data/foody/data' + str(dim) + '.p',
        report_to="../data/foody/report_dscnn",
        batch_size=20,
        valid_fold=0,
        max_epochs=40,
        filter_hs=[3, 5, 7],
        feature_maps=10
    )
