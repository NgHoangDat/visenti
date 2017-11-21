import sys
sys.path.append("..")

from six.moves import cPickle

import viprocessor as vp
import conv_net_sentence as cns

def main(data_folder, data_file, we_folder, dim=300, cv=10, non_static=True, we="rand"):
    cPickle.dump(vp.build_dataset(data_folder, we_folder, dim, cv), open(data_file, "wb"))
    cns.train(data_file, cv, non_static, we)


if __name__ == '__main__':
    data_folder = [
        "../data/en-neg",
        "../data/en-pos"
    ]
    data_file = "../data/data.p"
    we_folder = "../data/"

    dim = 100
    cv = 10

    non_static = sys.argv[1] == "-nonstatic"
    we = sys.argv[2][1:]
    
    main(
        data_folder=data_folder,
        data_file=data_file,
        we_folder=we_folder,
        dim=dim,
        cv=cv,
        non_static=non_static,
        we=we
    )