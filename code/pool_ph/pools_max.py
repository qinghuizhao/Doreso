#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: pools_max.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-05 14:48
# ===================================================================

import numpy as np

def pools_max(mat, wsize = 100, hsize = 100):
    """ Pool the imput matrix with a fixed window size using max method.

    Inputs:
        mat     -- is the input matrix of audio files.
        wsize   -- is the size of pooling window(the number of frames).
        hsize   -- is the hop size of pooling operation(the number of frames).
    Outputs:
        the pooled matrix, smaller than input matrix.
    note:
        the input matrix may 
    """
    pooled_mat = []

    # calc the number of pooling windows
    wnum = mat.shape[0] / wsize

    # pool opreation
    s = 0; e = wsize
    for w in xrange(wnum - 1):
        pooled_mat.append(np.amax(mat[s:e, :], axis = 0))
        print np.amax(mat[s:e, :], axis =0)
        s = s + hsize  # start frame
        e = e + hsize  # end frame
    print 's,e:', s, e
    pooled_mat.append(np.amax(mat[s:, :], axis = 0))
    print np.amax(mat[s:, :], axis =0)

    # convert list to numpy.array
    pooled_mat = np.array(pooled_mat)

    return pooled_mat

if __name__ == '__main__':
    m = np.ones((1256, 120))
    m[0,:] = 2
    m[-1,:] = 2
    print 'm is:', m
    print 'm.shape:', m.shape
    a = pools_max(m,100,100)
    print a, a.shape
