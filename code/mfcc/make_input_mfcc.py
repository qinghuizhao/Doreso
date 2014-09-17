#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: make_input_mfcc.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-11 11:06
# ===================================================================

# make input data for mlp model

import numpy as np
import os

def make_mat(inpath):
    """
    make_mat() will make all the npy files into one numpy.array
    """
    lf = os.listdir(inpath)  # file list of input folder
    lf.sort()

    joined_mat = np.array([])
    label = np.array([], dtype = int)
    lab_index = 0
    for folder in lf:
        subdir = os.path.join(inpath, folder)
        sublf = os.listdir(subdir)
        sublf.sort()

        for file in sublf:
            filepath = os.path.join(subdir, file)
            cur = np.load(filepath)
            cur = subsampling(cur)
            joined_mat = np.concatenate([x for x in [joined_mat, cur] if x.size > 0])
            label = np.concatenate([x for x in [label, np.ones(len(cur)) * lab_index]\
                    if x.size > 0])

        lab_index = lab_index + 1

    return joined_mat, label

def subsampling(mat):
    index = range(0, 1000, 10)
    new = mat[index]
    return new

def make_input(inpath, outpath):
    """
    make_input() will make the *.npy files from input path into particular format,
    so that they can be used as the input file of MLP.

    Inputs:
        inpath  -- the input path of *.npy files.
        outpath -- the output path of formated file.

    the format of outputs: 
        list[numpy.array[trian], numpy.array[test], numpy.array[validation]]
    """
    tz_mfcc = []  # tzanetakis mfcc

    tr_mfcc, tr_lab = make_mat(inpath + '/train/')
    val_mfcc, val_lab = make_mat(inpath + '/validation/')
    te_mfcc, te_lab = make_mat(inpath + '/test/')

    tz_mfcc.append((tr_mfcc, tr_lab))
    tz_mfcc.append((val_mfcc, val_lab))
    tz_mfcc.append((te_mfcc, te_lab))

    np.save(outpath + '/tzanetakis_mfcc', tz_mfcc)


if __name__ == '__main__':
    inp = '/media/tiger/data/splited/Tzanetakis/mfcc_binary'
    outp = '/media/tiger/data/splited/Tzanetakis/mfcc_binary'
    make_input(inp, outp)
    print 'Finished...'
