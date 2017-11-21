import re
import os
import string
from collections import defaultdict
import sys
from six.moves import cPickle

import numpy as np
import pandas as pd
from nltk.data import load
from pyvi.pyvi import ViTokenizer
from gensim.models import Word2Vec

VN_SENT_MODEL = 'file:' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'vietnamese.pickle')


def separate_sentence(paragraph: str):
    tokenizer = load(VN_SENT_MODEL)
    return tokenizer.tokenize(paragraph)


def tokenize(sentence: str):
    return ViTokenizer.tokenize(input)


def remove_punctuation(text: str):
    return " ".join([w for w in text.split() if w not in set(string.punctuation)])


def read_file(data_folder: list):
    revs = []
    for i in range(len(data_folder)):
        with open(data_folder[i], 'r', encoding='utf-8') as f:
            revs += [(rev.split(), i) for rev in f.readlines()]
    return revs


def create_word_vec(revs: list, dim=300, min_count=5):
    wv = Word2Vec(revs, size=dim, min_count=min_count).wv
    return {w: wv[w] for w in wv.vocab}


def create_word_idx_map(vocab: list):
    i = 1
    word_idx_map = dict()
    for w in vocab:
        word_idx_map[w] = i
        i += 1 
    return word_idx_map


def get_W(word_vecs, dim=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    W = np.zeros(shape=(vocab_size + 1, dim), dtype='float32')
    W[0] = np.zeros(dim, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        i += 1
    return W


def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector. 0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, dim)  

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub("\<", " < ", string)
    string = re.sub(",", " , ", string)
    string = re.sub("!", " ! ", string)
    string = re.sub("\(", " \( ", string)
    string = re.sub("\)", " \) ", string)
    string = re.sub("\?", " \? ", string)
    string = re.sub("\s+", " ", string)
    return string.strip()


def build_data_cv(revs, cv=10, lowered=True, tokenized=True, punctuation_removed=True, cleaned=True):
    count = defaultdict(int)
    max_l = 0
    vocab = defaultdict(float)
    out_revs = []
    for r in revs:
        rev, label = r
        rev = " ".join(rev)

        rev = remove_punctuation(rev) if punctuation_removed else rev
        rev = rev.lower() if lowered else rev
        rev = tokenize(rev) if tokenized else rev
        rev = clean_str(rev) if cleaned else rev

        words = set(rev.split())
        for w in words:
            vocab[w] += 1

        num_words = len(rev.split())
        max_l = num_words if max_l < num_words else max_l
        count[str(label)] += 1

        out_revs.append({
            "y": label,
            "text": rev,
            "num_words": num_words,
            "split": np.random.randint(0, cv)
        })
    return out_revs, vocab, max_l, count


def build_dataset(data_folder, we_folder, dim=300, cv=10):
    print("loading data...")
    origin_revs = read_file(data_folder)
        
    revs, vocab, max_l, count = build_data_cv(
        origin_revs,
        cv=cv,
        lowered=True,
        tokenized=False,
        punctuation_removed=False,
        cleaned=True
    )
    print("data loaded!")
        
    print("number of sentences: " + str(len(revs)))
    print("number of rev for each class: ")
    for y in count:
        print("{0}: {1}".format(y, count[y]))

    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
        
    word_idx_map = create_word_idx_map(vocab)
    word_embeding = {}
        

    print("building  word2vec...")
    wv = create_word_vec([rev for rev, _ in origin_revs], dim=dim)
    add_unknown_words(wv, vocab, dim=dim)
    word_embeding['w2v'] = get_W(wv, dim=dim)
    
    print("building random word vector...")
    wv = {}
    add_unknown_words(wv, vocab, dim=dim)
    word_embeding['rand'] = get_W(wv, dim=dim)
    
    print("loading glove...")
    wv = {}
    print(we_folder)
    with open(we_folder + 'glove' + str(dim), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split()
            if line[0] in vocab:
                wv[line[0]] = np.array(line[1:]).astype(np.float32)
    add_unknown_words(wv, vocab, dim=dim)
    word_embeding['glove'] = get_W(wv, dim=dim)

    print("dataset created!")
    return revs, word_embeding, word_idx_map, vocab, max_l


if __name__=="__main__":  
    data_folder = [
        "./data/en-neg",
        "./data/en-pos"
    ]
    we_folder = './data/'
    save_file = "./data/data.p"
    dim = 300
    cv = 10
    cPickle.dump(build_dataset(data_folder, we_folder, dim, cv), open(save_file, "wb"))
