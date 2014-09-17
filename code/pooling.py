#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: pooling.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-09 15:39
# ===================================================================

import numpy as np

def _pools_max(mat, wsize = 100, hsize = 100):
    """ Pool the imput matrix with a fixed window size using max method.

    Inputs:
        mat     -- is the input matrix of audio files.
        wsize   -- is the size of pooling window(the number of frames).
        hsize   -- is the hop size of pooling operation(the number of frames).
    Outputs:
        the pooled matrix, smaller than input matrix.
    """
    pooled_mat = []

    # calc the number of pooling windows
    wnum = (mat.shape[0] - wsize) // hsize + 1

    # pool opreation
    s = 0; e = wsize
    for w in xrange(wnum - 1):
        pooled_mat.append(np.amax(mat[s:e, :], axis = 0))
        # print np.amax(mat[s:e, :], axis =0)
        s = s + hsize  # start frame
        e = e + hsize  # end frame
    # print 's,e:', s, e
    pooled_mat.append(np.amax(mat[s:, :], axis = 0))
    # print np.amax(mat[s:, :], axis =0)

    # convert list to numpy.array
    pooled_mat = np.array(pooled_mat)

    return pooled_mat
#    l = mat.shape[0]
#    for i in range(0,l,hopsize):
#	if i + hopsize < l:
#		pooled_mat.append(np.amax(mat[i:i+hopsize,:],axis = 0)) 
#
def _pools_mean(mat, wsize = 100, hsize = 100):
    """ Pool the imput matrix with a fixed window size using mean method.

    Inputs:
        mat     -- is the input matrix of audio files.
        wsize   -- is the size of pooling window(the number of frames).
        hsize   -- is the hop size of pooling operation(the number of frames).
    Outputs:
        the pooled matrix, smaller than input matrix.
    """
    pooled_mat = []

    # calc the number of pooling windows
    wnum = mat.shape[0] / wsize

    # pool opreation
    s = 0; e = wsize
    for w in xrange(wnum - 1):
        pooled_mat.append(np.mean(mat[s:e, :], axis = 0))
        # print np.mean(mat[s:e, :], axis = 0)
        s = s + hsize  # start frame
        e = e + hsize  # end frame
    # print 's,e:', s, e
    pooled_mat.append(np.mean(mat[e:, :], axis = 0))
    # print np.mean(mat[s:, :], axis = 0)

    # convert list to numpy.array
    pooled_mat = np.array(pooled_mat)

    return pooled_mat


def _pools_min(mat, wsize = 100, hsize = 100):
    """ Pool the imput matrix with a fixed window size using min method.

    Inputs:
        mat     -- is the input matrix of audio files.
        wsize   -- is the size of pooling window(the number of frames).
        hsize   -- is the hop size of pooling operation(the number of frames).
    Outputs:
        the pooled matrix, smaller than input matrix.
    """
    pooled_mat = []

    # calc the number of pooling windows
    wnum = (mat.shape[0] - wsize) / hsize

    # pool opreation
    s = 0; e = wsize
    for w in xrange(wnum - 1):
        pooled_mat.append(np.amin(mat[s:e, :], axis = 0))
        # print np.amin(mat[s:e, :], axis =0)
        s = s + hsize  # start frame
        e = e + hsize  # end frame
    # print 's,e:', s, e
    pooled_mat.append(np.amin(mat[s:, :], axis = 0))
    # print np.amin(mat[s:, :], axis =0)

    # convert list to numpy.array
    pooled_mat = np.array(pooled_mat)

    return pooled_mat


def _pools_var(mat, wsize = 100, hsize = 100):
    """ Pool the imput matrix with a fixed window size using variance method.

    Inputs:
        mat     -- is the input matrix of audio files.
        wsize   -- is the size of pooling window(the number of frames).
        hsize   -- is the hop size of pooling operation(the number of frames).
    Outputs:
        the pooled matrix, smaller than input matrix.
    """
    pooled_mat = []

    # calc the number of pooling windows
    wnum = mat.shape[0] / wsize

    # pool opreation
    s = 0; e = wsize
    for w in xrange(wnum - 1):
        pooled_mat.append(np.var(mat[s:e, :], axis = 0))
        # print np.var(mat[s:e, :], axis = 0)
        s = s + hsize  # start frame
        e = e + hsize  # end frame
    # print 's,e:', s, e
    pooled_mat.append(np.var(mat[s:, :], axis = 0))
    # print np.var(mat[s:, :], axis = 0)

    # convert list to numpy.array
    pooled_mat = np.array(pooled_mat)

    return pooled_mat


def pooling(inputmat, func, wsize = 100, hsize = 100):
    """
    select pooling functions, and pooling the input matrix.
    Inputs:
        inputmat    -- is the input matrix of audio file.
        func        -- is the pooling functions to be used, it can be one of
                       ('min', 'max', 'mean', 'var').
        wsize       -- is the size of pooling window(the number of frames).
        hsize       -- is the hop size of pooling operation(the number of frames).
    Outputs:
        the pooled matrix, smaller than input matrix.
    """
    if func == 'max':
        return _pools_max(inputmat, wsize, hsize)
    elif func == 'min':
        return _pools_max(inputmat, wsize, hsize)
    elif func == 'mean':
        return _pools_max(inputmat, wsize, hsize)
    elif func == 'var':
        return _pools_max(inputmat, wsize, hsize)

if __name__ == '__main__':
    m = np.ones((1256, 120))
    m[0,:] = 2
    m[-1,:] = 2
    print 'm is:', m
    print 'm.shape:', m.shape
    a = pooling(m, 'max', 100,100)
    b = pooling(m, 'min', 100,100)
    c = pooling(m, 'mean', 100,100)
    d = pooling(m, 'var', 100,100)
    print a, a.shape
    print b, b.shape
    print c, c.shape
    print d, d.shape
