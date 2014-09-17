# ===================================================================
#     FileName: triangular_filter.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-03 18:15
# ===================================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import log10

def trfcoef(fs, nfft, nfilt):
    """ Compute triangular filterbank.

    Inputs:
        fs      -- is the sampling fraquency (Hz).
        nfft    -- is the size of fft array.
        nfilt   -- is the number of filters.

    Outputs:
        T       -- are the triangular filters in a matrix.
    """

    # Compute start/middle/end points of the triangular filters in spectral domain.
    hfs = fs / 2
    max_mel = hz2mel(hfs)  # compute the max frequency in mel-scale.

    # calc the mel frequencies
    mfreqs = np.arange(nfilt + 2) * max_mel / (nfilt + 1)

    # tranform frequencies in mel to frequencies in Hz.
    freqs = mel2hz(mfreqs)
    heights = 1. / (freqs[1 :] - freqs[: -1])

    # Compute filter's coeff (in fft domain, in bins)
    fcoef = np.zeros((nfilt, nfft))

    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * hfs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        # calc the left ids
        lid = np.arange(np.floor(low * nfft / hfs) + 1,
                        np.floor(cen * nfft / hfs) + 1, dtype=np.int)
        lslope = heights[i]

        # calc the right ids
        rid = np.arange(np.floor(cen * nfft / hfs) + 1,
                        np.floor(hi * nfft / hfs) + 1, dtype=np.int)
        rslope = heights[i + 1]

        fcoef[i][lid] = lslope * (nfreqs[lid] - low)
        fcoef[i][rid] = rslope * (hi - nfreqs[rid])

    return fcoef.T

def hz2mel(f):
    return 2585 * log10(1 + f / 700)

def mel2hz(m):
    return (pow(10, m / 2595) - 1) * 700
