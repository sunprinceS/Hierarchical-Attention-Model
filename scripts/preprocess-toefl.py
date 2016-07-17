#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing script for TOEFL listening comprehension
"""

import os
import glob
import sys
import argparse
from utils import build_vocab,get_sentence,make_dirs,dependency_parse

def preprocess_story(data_dir):
    queries=[]
    choices=[]
    sents=[]
    num_sent_table=[]
    labels=[]
    answers = []
    file_list = [f for f in os.listdir(data_dir['src']) \
            if os.path.isfile(os.path.join(data_dir['src'],f))]

    for file in file_list:
        with open(os.path.join(data_dir['src'],file)) as story_file:
            opinion_cnt = 0
            sent_cnt = 0
            cnt = 0
            label=[]
            answer = []
            try:
                story = story_file.read().splitlines()
                for line in story:
                    sentence_type,sentence = get_sentence(line)
                    if sentence_type == 'SENTENCE':
                        sent_cnt += 1
                        sents.append(' '.join(sentence))
                    elif sentence_type == 'QUESTION':
                        queries.append(' '.join(sentence))
                    else: #OPINION
                        opinion_cnt += 1
                        choices.append(' '.join(sentence[:-2]))
                        if sentence[-2] == '1':
                            cnt += 1
                            answer.append(opinion_cnt)
                        label.append(float(sentence[-2]))
                label = [x/cnt for x in label]
                labels.append(' '.join(format(x,"10.2f") for x in label))
                answers.append(' '.join(map(str,answer)))
                num_sent_table.append(str(sent_cnt))
            except UnicodeDecodeError:
                pass

    with open(os.path.join(data_dir['des'],'queries'),'w') as queries_file, \
         open(os.path.join(data_dir['des'],'num_sent'),'w') as num_sent_file, \
         open(os.path.join(data_dir['des'],'labels'),'w') as label_file, \
         open(os.path.join(data_dir['des'],'answers'),'w') as answer_file, \
         open(os.path.join(data_dir['des'],'sents'),'w') as sents_file, \
         open(os.path.join(data_dir['des'],'choices'),'w') as choices_file:
        queries_file.write('\n'.join(queries))
        num_sent_file.write('\n'.join(num_sent_table))
        label_file.write('\n'.join(labels))
        answer_file.write('\n'.join(answers))
        sents_file.write('\n'.join(sents))
        choices_file.write('\n'.join(choices))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='preprocess-toefl.py',description='Preprocess the toefl dataset.')
    # parser.add_argument('--datatype',type=str,default='manual',choices=['manual','ASR'])
    # args = parser.parse_args()
    print('=' * 80)
    print('Preprocessing TOEFL listening comprehension dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lib_dir = os.path.join(base_dir, 'lib')

    sent_paths = list()
    query_paths = list()
    choice_paths = list()
    data_types = ['manual_trans','ASR_trans']

    for data_type in data_types:
        orig_data_dir = os.path.join(base_dir,'to_project',data_type)
        toefltask_dir = os.path.join(data_dir, 'toefl',data_type)

        train_dir=dict()
        dev_dir=dict()
        test_dir=dict()

        train_dir['des'] = os.path.join(toefltask_dir, 'train')
        dev_dir['des'] = os.path.join(toefltask_dir,'dev')
        test_dir['des'] = os.path.join(toefltask_dir ,'test')

        train_dir['src'] = os.path.join(orig_data_dir, 'train')
        dev_dir['src'] = os.path.join(orig_data_dir,'dev')
        test_dir['src'] = os.path.join(orig_data_dir ,'test')

        if data_type == 'manual_trans':
            traverse_dir = [train_dir,dev_dir,test_dir]
        else: #ASR_trans
            traverse_dir = [test_dir]

        for d in traverse_dir:
            preprocess_story(d)

        sent_paths.extend(glob.glob(os.path.join(toefltask_dir, '*/sents')))
        query_paths.extend(glob.glob(os.path.join(toefltask_dir ,'*/queries')))
        choice_paths.extend(glob.glob(os.path.join(toefltask_dir,'*/choices')))

    # produce dependency parses
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])
    for filepath in sent_paths:
        dependency_parse(filepath,cp=classpath, tokenize=False,sent_type='sent')
    for filepath in query_paths:
        dependency_parse(filepath,cp=classpath, tokenize=False,sent_type='query')
    for filepath in choice_paths:
        dependency_parse(filepath,cp=classpath, tokenize=False,sent_type='choice')

    #generate whole vocabulary
    for data_type in data_types:
        toefltask_dir = os.path.join(data_dir, 'toefl',data_type)
        build_vocab(sent_paths+query_paths+choice_paths, os.path.join(toefltask_dir, 'vocab.txt'))
        build_vocab(sent_paths+query_paths+choice_paths, os.path.join(toefltask_dir, 'vocab-cased.txt'), lowercase=False)

