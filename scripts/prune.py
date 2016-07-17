import numpy as np
import pickle
import joblib
import sys
import os
from scipy.spatial.distance import cdist
from utils import LoadGloVe, GetWordFeature

proto=pickle.HIGHEST_PROTOCOL
emb = np.zeros((2196017,300),dtype='float32')

def getSentFeature(sent, emb, vocab_map):
    feat = np.zeros((300), float)
    for word in sent:
        feat += GetWordFeature(word, emb, vocab_map)
    return feat

def writeSent(sent, file):
    for i in range(len(sent)):
        word = sent[i]
        if i == len(sent) - 1:
            file.write(word + '\n')
        else:
            file.write(word + ' ')

emb, vocab_map = LoadGloVe()

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
toefl_manual_dir = os.path.join(base_dir,'data','toefl','manual_trans')
toefl_ASR_dir = os.path.join(base_dir,'data','toefl','ASR_trans')

dir_ls = list()
for sub_dir in ['train','dev','test']:
    dir_ls.append(os.path.join(toefl_manual_dir,sub_dir))
dir_ls.append(os.path.join(toefl_ASR_dir,'test'))

frac = float(sys.argv[1])

for file_dir in dir_ls:
    with open(os.path.join(file_dir,'num_sent'), 'r') as nsent_file, \
         open(os.path.join(file_dir,'sent.toks'), 'r') as sent_file, \
         open(os.path.join(file_dir,'query.toks'), 'r') as query_file, \
         open(os.path.join(file_dir,'sents_%.1f'%(frac)), 'w') as pruned_file:
        sents = []
        n_sents = []
        queries = []
        new_sents = []
        for sent in sent_file:
            sents.append(sent.strip().split())
        for nsent in nsent_file:
            n_sents.append(int(nsent.strip()))
        for query in query_file:
            queries.append(query.strip().split())

        sent_file.seek(0)

        idx = 0
        for (query, nsent) in zip(queries, n_sents):
            if nsent == 0:
                continue
            # get query feature
            query_feat = np.reshape(getSentFeature(query, emb, vocab_map), [1,300])
            # get features for each sentence in the document
            for i in range(nsent):
                if i == 0:
                    sents_feat = getSentFeature(sents[idx+i], emb, vocab_map)
                    if nsent == 1:
                        sents_feat = np.reshape(sents_feat, [1,300])
                else:
                    new_feat = getSentFeature(sents[idx+i], emb, vocab_map)
                    sents_feat = np.vstack([sents_feat, new_feat])
            # calculate similarity
            dist = cdist(query_feat, sents_feat, 'cosine')
            # ranking
            sorted_idx = np.argsort(dist, axis=1)
            # number of sentence kept
            n_keep = np.ceil(nsent * frac)
            # check and write sentences to file
            for i in range(np.shape(sorted_idx)[1]):
                if sorted_idx[0,i] < n_keep:
                    writeSent(sents[idx+i], pruned_file)
            idx += nsent
