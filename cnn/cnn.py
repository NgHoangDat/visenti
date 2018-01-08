import sys
sys.path.append("..")
# import argparse

from six.moves import cPickle

import viprocessor as vp
import conv_net_sentence as cns

def main(data_folder, data_file, we_folder, report_to, rebuild=False, limit=None, dim=300, cv=10, non_static=True, we="rand", batch_size=50):
    if rebuild:
        cPickle.dump(vp.build_dataset(
            data_folder=data_folder,
            we_folder=we_folder,
            dim=dim,
            cv=cv,
            limit=limit
        ), open(data_file, "wb"))
    
    cns.train(
        data_file=data_file,
        valid_fold=0,
        report_file=report_to,
        non_static=non_static,
        we=we,
        batch_size=50
    )


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument('-nonstatic', type=int, default=1)
    #parser.add_argument('-pos', type=str, default="../data/foody/pos")
    #parser.add_argument('-neg', type=str, default="../data/foody/neg")
    #parser.add_argument('-saveto', type=str,  default="../data/foody/data25.p")
    # parser.add_argument('-report_to', type=str,  default="../data/foody/report_cnn")
    #parser.add_argument('-wef', type=str,  default="../data/foody/")
    #parser.add_argument('-rebuild', type=int, default=0)
    # parser.add_argument('-we', type=str,  default="w2v")
    # parser.add_argument('-dim', type=int,  default=300)
    #parser.add_argument('-cv', type=int, default=10)
    #parser.add_argument('-limit', type=int, default=None)
    # parser.add_argument('-bsize', type=int, default=20)
    # args = vars(parser.parse_args())
    #data_folder = [
    #    args['neg'],
    #    args['pos']
    #]
    
    #main(
    #    data_folder=data_folder,
    #    data_file=args['saveto'],
    #    we_folder=args['wef'],
    #    report_to=args['report_to'],
    #    limit=args['limit'],
    #    dim=args['dim'],
    #    cv=args['cv'],
    #    non_static=args['nonstatic'] == 1,
    #    we=args['we'],
    #    batch_size=args['bsize']
    #)
    # we = sys.argv[1]
    # options = [
    #    [10, 2],
    #    [20, 2],
    #    [40, 2],
    #    [80, 2]
    #]
    dim = 100
    # i = sys.argv[1]
    data_file = '../data/foody/data' + str(dim) + '.p'
    cns.train(
        data_file=data_file,
        valid_fold=0,
        report_file="../data/foody/report_cnn",
        non_static=True,
        filter_hs=[3, 5, 7], 
        hidden_units=[40, 2],
        we='w2v',
        batch_size=20
    )
