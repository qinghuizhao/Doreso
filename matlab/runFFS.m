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

[new_feature,new_acc,pre_acc] = FFS(train_mfcc,train_HPSS,train_labels,test_mfcc,test_HPSS,test_labels)


