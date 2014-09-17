#!/bin/bash

# prepare the data
	cat lst/train.list > lst/UBM.lst
	python GenTrain.py lst/train.list > ndx/trainModel.ndx

# training the UBM
	echo "training the UBM"
	bin/TrainWorld --config cfg/TrainWorld.cfg &> log/TrainWorld.log

# training the GMM
	echo "Training the GMM"
	bin/TrainTarget --config cfg/TrainTarget.cfg &> log/TrainTarget.cfg



# 1.generate training liklihood
	echo "generating the liklihood of training data"
	python GenTest.py lst/train.list > ndx/computetest_gmm_target-seg.ndx
	bin/ComputeTest --config cfg/ComputeTest_GMM.cfg &> log/ComputeTest.cfg
	python CalcRes.py res/target-seg_gmm.res
	python ProcessRes.py res/target-seg_gmm.res > ~/tiger/tools/XZ/libsvm/matlab/data/train.txt

#2.generate test liklihood
	echo "generating the liklihook of test data"
	python GenTest.py lst/test.list > ndx/computetest_gmm_target-seg.ndx
	bin/ComputeTest --config cfg/ComputeTest_GMM.cfg &> log/ComputeTest.cfg
	python CalcRes.py res/target-seg_gmm.res
	python ProcessRes.py res/target-seg_gmm.res > ~/tiger/tools/XZ/libsvm/matlab/data/test.txt


