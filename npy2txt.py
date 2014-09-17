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

def make_mat(inpath,outpath):
    """
    make_mat() will make all the npy files into one numpy.array
    """
    lf = os.listdir(inpath)  # file list of input folder
    lf.sort()
#
#    joined_mat = np.array([])
    label = np.array([], dtype = int)
    lab_index = 1
    flag = False
    for folder in lf:
        subdir = os.path.join(inpath, folder)
        sublf = os.listdir(subdir)
        sublf.sort()
	out_subdir = os.path.join(outpath,folder)

        for file in sublf:
            filepath = os.path.join(subdir, file)
	    #a,b,c = os.path.split(file)
    	    if not os.path.exists(out_subdir):
		os.makedirs(out_subdir)	    
	    filename = os.path.join(out_subdir,file)
	    a,b = os.path.splitext(filename)
            
	    cur = np.load(filepath)
	    np.savetxt(a + '.txt',cur, fmt = '%d', delimiter=' ',newline='\n')
     #       joined_mat = np.concatenate([x for x in [joined_mat, cur] if x.size > 0])
	   # label = np.concatenate([x for x in [label, lab_index] if x <= 10 and x >= 0] )
            label = np.append(label,lab_index)
        lab_index = lab_index + 1

    return  label

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
        list[numpy.array[trian], numpy.array[train], numpy.array[train]]
    """
    tz_mfcc = []  # tzanetakis mfcc

    te_lab = make_mat(inpath + '/train/',outpath + '/train/')

    #tz_mfcc.append((te_mfcc, te_lab))

    np.savetxt(outpath + '/label.txt', te_lab, delimiter=' ', newline = '\n')


if __name__ == '__main__':
    inp = '/media/tiger/data/out/Prime/HPSS_MFCC'
    outp = '/media/tiger/data/FeatureTxt/Prime/HPSS_MFCC/trainset_splited'
    if os.path.exists(outp):
        os.system('rm -r '+ outp)

   
    make_input(inp, outp)
    print 'Finished...'
