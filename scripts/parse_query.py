#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
lib_dir = os.path.join(base_dir, 'lib')

toefl_manual_dir = os.path.join(base_dir,'data','toefl','manual_trans')
toefl_ASR_dir = os.path.join(base_dir,'data','toefl','ASR_trans')

query_dir_ls = list()
for sub_dir in ['train','dev','test']:
    query_dir_ls.append(os.path.join(toefl_manual_dir,sub_dir))
query_dir_ls.append(os.path.join(toefl_ASR_dir,'test'))

classpath = ':'.join([
    lib_dir,
    os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
    os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

for q_dir in query_dir_ls:
    q = os.path.join(q_dir,'queries_symbol_sep')

    queries = list()
    with open(q,'r') as query_file:
        queries = query_file.read().splitlines()

    query_sent = list()
    query_map = list()

    for idx,query in enumerate(queries,1):
        sep_idx = query.find('|')
        if sep_idx != -1:
            query_sent.append(query[:sep_idx])
            query_map.append(str(idx))
            query_sent.append(query[sep_idx+1:])
            query_map.append(str(idx))
        else:
            query_sent.append(query)
            query_map.append(str(idx))

    with open(os.path.join(q_dir,'query_id.map'),'w') as query_id_file, \
         open(os.path.join(q_dir,'queries_sep'),'w') as query_file:
        query_id_file.write('\n'.join(query_map))
        query_file.write('\n'.join(query_sent))

    filename = os.path.basename(q)
    print('\nDependency parsing ' + filename)
    tokpath = os.path.join(q+'.toks')
    parentpath = os.path.join(q_dir,'query_sep_dparents')

    tokenize_flag = '-tokenize - '
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath .jizz %s < %s'
        % (classpath, tokpath, parentpath, tokenize_flag, os.path.join(q_dir,'queries_sep')))
    os.system(cmd)
