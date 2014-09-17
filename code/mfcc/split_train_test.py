#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: split_train_test.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-09 17:18
# ===================================================================

import os
import random
def split(inpath, outpath, r_tr, r_te, r_val):
    for folder in os.listdir(inpath):
        subdir = os.path.join(inpath, folder)
        sublist = os.listdir(subdir)
        os.system('mkdir -p ' + outpath + '/train/' + folder)
        os.system('mkdir -p ' + outpath + '/test/' + folder)
        os.system('mkdir -p ' + outpath + '/validation/' + folder)
        # train
        for i in xrange(int(100 * r_tr)):
            file = random.choice(sublist)
            sublist.remove(file)
            os.system('cp ' + subdir + '/' + file + ' ' + outpath + '/train/' + folder)
        # test
        for i in xrange(int(100 * r_te)):
            file = random.choice(sublist)
            sublist.remove(file)
            os.system('cp ' + subdir + '/' + file + ' ' + outpath + '/test/' + folder)
        # validation
        for i in xrange(int(100 * r_val)):
            file = random.choice(sublist)
            sublist.remove(file)
            os.system('cp ' + subdir + '/' + file + ' ' + outpath + '/validation/' + folder)


if __name__ == '__main__':
    inp = '/media/tiger/data/out/Tzanetakis/mfcc'
    outp = '/media/tiger/data/splited/Tzanetakis/mfcc'
    # 1
    tr = 0.5
    te = 0.3
    val = 0.2
    # 2
    # tr = 0.8
    # te = 0.1
    # val = 0.1
    os.system('rm -r ' + outp)
    split(inp, outp, tr, te, val)
    print 'Finished!'
