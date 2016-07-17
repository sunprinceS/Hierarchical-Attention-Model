#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
lib_dir = os.path.join(base_dir, 'lib')
toefl_manual_dir = os.path.join(base_dir,'data','toefl','manual_trans')
toefl_ASR_dir = os.path.join(base_dir,'data','toefl','ASR_trans')

classpath = ':'.join([
    lib_dir,
    os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
    os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

dir_ls = list()
for sub_dir in ['train','dev','test']:
    dir_ls.append(os.path.join(toefl_manual_dir,sub_dir))
dir_ls.append(os.path.join(toefl_ASR_dir,'test'))

print('\nDependency parsing ' + sys.argv[1])
# dirpath = os.path.dirname(sys.argv[1])
for dirpath in dir_ls:
    tokpath = os.path.join(dirpath, 'sents_'+sys.argv[1] + '.toks')
    parentpath = os.path.join(dirpath,'sents_'+sys.argv[1]+'_dparents')

    tokenize_flag = '-tokenize - '
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath 12 %s < %s'
        % (classpath, tokpath, parentpath, tokenize_flag, os.path.join(dirpath,'sents_'+sys.argv[1])))
    os.system(cmd)
