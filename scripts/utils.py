#########################################################
#   FileName:       [ utils.py ]                        #
#   PackageName:    [ LSTM-MLP ]                        #
#   Synopsis:       [ Define util functions ]           #
#   Author:         [ MedusaLafayetteDecorusSchiesse]   #
#########################################################

import numpy as np
import scipy.io as sio
import joblib
import sys
import os
from itertools import zip_longest



#########################
#     I/O functions     #
#########################

def LoadGloVe():
    # output:
    #     word_embedding: a numpy array of shape (n_words, word_vec_dim), where n_words = 2196017 and word_vec_dim = 300
    #     word_map: a dictionary that maps words (strings) to their indices in the word embedding matrix (word_embedding)
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    glove_dir = os.path.join(base_dir,'data','glove')
    # xx = os.path.join(base_dir,'data','glove','glove.840B.300d.txt')
    word_embedding_ls = list()
    word_embedding = joblib.load(os.path.join(glove_dir,'glove.840B.300d.emb'))
    unk = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_embedding, unk])
    word_map = {}
    with open(os.path.join(glove_dir,'glove.840B.vocab'), 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            line = line.strip()
            word_map[line] = i
            i += 1
    return word_embedding, word_map


############################
#  Other Helper Functions  #
############################

def GetWordFeature(word, word_embedding, word_map):
    feature = np.zeros((300), float)
    if word in word_map:
        feature = word_embedding[word_map[word]]
    else:
        #feature = np.mean(word_embedding, axis=0)
        feature = word_embedding[word_embedding.shape[0]-1]
    return feature

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def get_sentence(sentence):
    s = sentence.split(' ')
    s_type = s[0]
    s = s[1:]
    return s_type,s

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def dependency_parse(filepath, cp='', tokenize=True,sent_type=''):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    # filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, sent_type + '.toks')
    parentpath = os.path.join(dirpath,'{}_dparents'.format(sent_type))

    relpath =  os.path.join(dirpath, '{}_rels'.format(sent_type))
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)
