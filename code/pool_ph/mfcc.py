#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: triangular_filter.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-03 14:10
# ===================================================================

import essentia
import essentia.standard as es
import numpy

# 声明需要用到的函数类别
loader = es.MonoLoader(filename = '../data/sound1.mp3')
w = es.Windowing(type = 'hann')
spectrum = es.Spectrum()
mfcc = MFCC()

mfccs = []
audio = loader()
for frame in es.FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    mfccs.append(mfcc_coeffs)

mfccs = es.array(mfccs).T
