#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: extract_all_mfcc.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-11 10:59
# ===================================================================

import os
import mfcc
import sys

def walk_dir(inpath, outpath):
    """
    walk_dir(inpath, outpath) will wlak the input path recursively, and output
    processed files into output path.

    Inputs:
        inpath  -- is the input path of raw files.
        outpath -- is the output path for processed files.
    Outputs:
        processed files in binary
    """
    # make the corresponding output path
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for item in os.listdir(inpath):
        ipath = os.path.join(inpath, item)  # subinpath
        opath = os.path.join(outpath, item)  # suboutpath

        if os.path.isdir(ipath):
            walk_dir(ipath, opath)
        elif os.path.splitext(ipath)[-1] == '.wav':
            print 'extracting', ipath
            mfcc.extract(ipath, opath)

if __name__ == '__main__':
    inp = sys.argv[1]
    outp = sys.argv[2]
    if os.path.exists(outp):
        os.system( 'rm -r '+ outp)
    walk_dir(inp, outp)

#    inp = '/media/tiger/data/splited/Prime/HPSS/percussive/train'
#    outp = '/media/tiger/data/out/Prime/HPSS_MFCC/train/'
#    if os.path.exists(outp):
#        os.system( 'rm -r '+ outp)
#    walk_dir(inp, outp)

