import numpy as np
from six.moves import cPickle
from collections import defaultdict
import sys, re
import pandas as pd


def build_data_cv(data_folder, cv=10, clean_string=True, max_l=100):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    pos_count = 0
    neg_file = data_folder[1]
    neg_count = 0
    vocab = defaultdict(float)
    with open(pos_file, "r", encoding="utf-8") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            if len(orig_rev.split()) > max_l:
                continue
            for word in words:
                vocab[word] += 1
            datum  = {
                "y":1,
                "text": orig_rev,
                "num_words": len(orig_rev.split()),
                "split": np.random.randint(0,cv)
            }
            pos_count += 1
            revs.append(datum)
    with open(neg_file, "r", encoding="utf-8") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            if len(orig_rev.split()) > max_l:
                continue
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {
                "y":0,
                "text": orig_rev,
                "num_words": len(orig_rev.split()),
                "split": np.random.randint(0,cv)
            }
            neg_count += 1
            revs.append(datum)
    return revs, vocab, pos_count, neg_count

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector. 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    #string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub("\'s", " \'s", string)
    #string = re.sub("\'ve", " \'ve", string)
    #string = re.sub("n\'t", " n\'t", string)
    #string = re.sub("\'re", " \'re", string)
    #string = re.sub("\'d", " \'d", string)
    #string = re.sub("\'ll", " \'ll", string)
    string = re.sub("\<", " < ", string)
    string = re.sub(",", " , ", string)
    string = re.sub("!", " ! ", string)
    string = re.sub("\(", " \( ", string)
    string = re.sub("\)", " \) ", string)
    string = re.sub("\?", " \? ", string)
    string = re.sub("\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

if __name__=="__main__":    
    #w2v_file = sys.argv[1]
    data_folder = ["G:/Desktop/visenti/data/en-pos","G:/Desktop/visenti/data/en-neg"]
    print("loading data...")
    revs, vocab, pos_count, neg_count = build_data_cv(data_folder, cv=10, clean_string=True, max_l=100)
    max_l = 100
    print("data loaded!")
    print("number of sentences: " + str(len(revs)))
    print("number of pos rev: " + str(pos_count))
    print("number of neg rev: " + str(neg_count))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    #print("loading word2vec vectors...")
    #w2v = load_bin_vec(w2v_file, vocab)
    #print("word2vec loaded!")
    #print("num words already in word2vec: " + str(len(w2v)))
    #add_unknown_words(w2v, vocab)
    #W = []
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, k=300)
    W2, word_idx_map = get_W(rand_vecs, k=300)
    cPickle.dump([revs, None, W2, word_idx_map, vocab], open("G:/Desktop/visenti/data//en-all.p", "wb"))
    print("dataset created!")
