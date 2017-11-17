import sys
sys.path.append("..")

from six.moves import cPickle

import viprocessor as vp
import conv_net_sentence as cns

def main(data_folder, data_file, dim=300, cv=10, non_static=True, use_randWV=True):
    cPickle.dump(vp.build_dataset(data_folder, dim, cv), open(data_file, "wb"))
    cns.train(data_file, cv, non_static, use_randWV)


if __name__ == '__main__':
    data_folder = [
        "../data/en-neg",
        "../data/en-pos"
    ]
    data_file = "../data/data.p"

    dim = 100
    cv = 10

    non_static = sys.argv[1] == "-nonstatic"
    use_randWV = sys.argv[2] == "-rand"
    
    main(data_folder, data_file, dim, cv, non_static, use_randWV)