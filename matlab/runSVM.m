clear all;
addpath('./helper_function/');
addpath('./data/');
%
load('HPSS_mfcc_info');
%train_HPSS = train_mfcc_x;
%test_HPSS = test_mfcc_x;
%validation_HPSS = validation_mfcc_x;
%
%load('mfcc_prime_hpss_info');
train_HPSS = train_mfcc_x;
test_HPSS = test_mfcc_x;


load('mfcc_pandora_info');
train_mfcc = train_mfcc_x;
test_mfcc = test_mfcc_x;

clear train_mfcc_x;
clear test_mfcc_x;
%validation_mfcc = validation_mfcc_x;
%
train_labels = train_mfcc_y;
test_labels = test_mfcc_y;
%validation_labels = validation_mfcc_y;
%[train_mfcc,train_HPSS,validation_mfcc,validation_HPSS,train_labels,validation_labels] = splitTri_Val(train_mfcc,train_HPSS,train_labels);

%load('sp_pca');
%[cf,cr] = SVM_fusion_stack(train_mfcc,train_HPSS,train_labels,validation_mfcc,validation_HPSS,...
%            validation_labels,test_mfcc,test_HPSS,test_labels,128,64)
%[cf,cr] = SVM_fusion_stack2(train_mfcc,train_labels,validation_mfcc,...
%            validation_labels,test_mfcc,test_labels,128,32)
% stack2 = 1



%load('lfp_blocksize_pca');


%
%cr = []
%for thres = 0.3
   
[cf,CR] = SVM_feature_fusion(train_mfcc,train_HPSS,train_labels,test_mfcc,test_HPSS,test_labels,128,64,1,0)
%feature_fusion64 
%cr = [cr,CR];
%end
%cr
%
%CR = SVM_aggregate_onset('mfccP_20_info')
%CR = SVM_aggregate('HPSS_mfcc_info') 
%
%cr = [];
% clear train_mfcc; clear test_mfcc; clear train_HPSS; clear test_HPSS;
% [cf,cr] = SVM_GMM_version1('train.txt','test.txt',train_labels,test_labels)

%  for alpha = 0.5:0.1:1.0
%    [res,CR] = SVM_classifier_fusion_new(train_mfcc,train_HPSS,train_labels,test_mfcc,test_HPSS,test_labels,128,32,3)
%    [res,CR] = SVM_classifier_fusion_new(train_mfcc,train_HPSS,train_labels,test_mfcc,test_HPSS,test_labels,128,32,3)
%     classifer_fusion = 1
%    SVM_classifier_fus = 1

%     cr = [cr,CR]
%  end
%cr
%%CR = SVM_classifier_fusion('~/tiger/data/splited/PandoraSubset/mfcc/trainset_splited/',...
%          '~/tiger/data/splited/PandoraSubset/mfcc/testset_splited/',128,32,0.0)

%cr = SVM_2_class(train_HPSS,train_labels,test_HPSS,test_labels)
%[cr,model] = SVM_2_class_sp(train_sp,train_labels,test_sp,test_labels)
%two_class 
%[cf,cr] = SVM_lfp_songlevel('mfccP_20_info',256,64)
%[cf,cr] = SVM_mfcc_songlevel('mfccP_20_info',512,64)
