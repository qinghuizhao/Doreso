#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: pmsc.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-04 11:29
# ===================================================================

import numpy as np
import essentia.standard as es
import trgfilter as tf

def pmsc(fname, outpath, fs = 22050, fsize = 1024, hsize = 512):
    """
    pmsc(fname, outpath, fs, fsize, hsize) will compute the pmsc of Audio file fname.

    Inputs:
        fname   -- is the name of audio file.
        outpath -- is the output path of processed files.
        fs      -- is the sampling frequency (Hz).
        fsize   -- is the size of each frame.
        hsize   -- is the hop size betwean frames.
    Outputs:
        the file contains the pmsc of audio file.
        in what format???
    """
    loader = es.MonoLoader(filename = fname, sampleRate = fs)
    w = es.Windowing(type = 'hann')
    spectrum = es.Spectrum()

    energy = []
    audio = loader()

    # *** compute spectrum of audio file ***
    # this will generate more frames.
    for frame in es.FrameGenerator(audio, fsize, hsize):
        energy.append(spectrum(w(frame)))

    # this will abandoned the tail of audio
    # for fstart in range(0, len(audio) - fsize, hsize):
        # frame = audio[fstart : fstart + fsize]
        # energy.append(spectrum(w(frame)))

    # convert list to np.array
    energy = np.array(energy)
    print 'audio:', audio.size, 'energy size:', energy.size, 'energy shape:', energy.shape

    # *** mel compression ***
    # cacl the coefficents matrix of triangular filters
    tri_filters = tf.trfcoef(fs, energy.shape[1], 128)

    # conduct compression
    meled = np.dot(energy, tri_filters)  # meled matrix
    print 'mel shape', meled.shape
    print 'mel type', type(meled)

    # *** pca whitening ***
    # calc variances of meled
    variances = np.var(meled, axis = 0)
    print 'var shape', variances.shape

    # get the index of first eight variances and whitening
    index = np.argsort(variances)[:8]
    meled = np.delete(meled, index, axis = 1)
    print 'index', index
    print 'meled shape', meled.shape

    # multiply each component by the inverse square of its eigenvalue to obtain features
    # with unitary variance.
    invar = 1. / variances ** 2  # the inverse square of variances
    print 'var shape is:', invar.shape
    invar = np.delete(invar, index, axis = 0)
    print 'var shape is:', invar.shape
    meled = meled * invar

    np.save(outpath, meled)
    print 'save file to', (outpath, '.npy')

if __name__ == '__main__':
    print 'processing sound 1, runing ...'
    pmsc('../../data/sound1.mp3', './sound1.mp3')
