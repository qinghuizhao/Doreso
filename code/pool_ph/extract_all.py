#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: extract_all.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-09 11:23
# ===================================================================

import os
import pmsc

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
    os.makedirs(outpath)

    for item in os.listdir(inpath):
        ipath = os.path.join(inpath, item)  # subinpath
        opath = os.path.join(outpath, item)  # suboutpath

        if os.path.isdir(ipath):
            walk_dir(ipath, opath)
        elif os.path.splitext(ipath)[1] == '.mp3':
            pmsc.pmsc(ipath, opath)

if __name__ == '__main__':
    walk_dir('../../data/', '../../out/')
