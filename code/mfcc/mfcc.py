#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: triangular_filter.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-03 14:10
# ===================================================================

import essentia as e
import essentia.standard as es
import numpy as np
import sys
from wavread import gate

def extract(fname, outpath, fs = 22050, fsize = 1024, hsize = 512):
    """
    extract(fname, outpath, fs, fsize, hsize) will compute the mfcc of Audio file fname.

    Inputs:
        fname   -- is the name of audio file.
        outpath -- is the output path of processed files.
        fs      -- is the sampling frequency (Hz).
        fsize   -- is the size of each frame.
        hsize   -- is the hop size betwean frames.
    Outputs:
        the file contains the mfcc coefficents of audio file.
        in what format???
    """
#    gate(fname)    
    loader = es.MonoLoader(filename = fname, sampleRate = fs)
#    length = len(loader)
#    maxim = max(loader)
#    for sample in loader:
#        if abs(sample) < maxim/20:
#            sample = 0 ;
    
    w = es.Windowing(type = 'hann')
    spectrum = es.Spectrum()
    mfcc = es.MFCC(inputSize = 513, numberCoefficients = 20)

    mfccs = []
    audio = loader()
    for frame in es.FrameGenerator(audio, frameSize = 1024, hopSize = 512):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)

    mfccs = np.array(mfccs)
    return mfcc
    #print mfccs.shape
    #np.save(outpath, mfccs)
    #print 'save file to', outpath + '.npy'

if __name__ == '__main__':
    soundfile = sys.argv[1]
    outp = sys.argv[2]
    extract(soundfile, outp, fs)
