#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===================================================================
#     FileName: make_svm_mfcc.py
#       Author: Beanocean
#        Email: beanocean@outlook.com
#   CreateTime: 2014-06-11 19:48
# ===================================================================

import numpy as np
path = '/media/tiger/data/splited/Tzanetakis/mfcc_binary'
mfcc = np.load(path + '/tzanetakis_mfcc.npy')
np.savetxt(path + '/tz_mfcc_train.txt', mfcc[0][0])
np.savetxt(path + '/tz_mfcc_trlab.txt', mfcc[0][1])
np.savetxt(path + '/tz_mfcc_test.txt', mfcc[2][0])
np.savetxt(path + '/tz_mfcc_telab.txt', mfcc[2][1])
