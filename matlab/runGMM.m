
clear all;
addpath('./helper_function/');
addpath('./data/');

load('mfcc_pandora_info');
train_mfcc = train_mfcc_x;
test_mfcc = test_mfcc_x;

clear train_mfcc_x;
clear test_mfcc_x;
%validation_mfcc = validation_mfcc_x;
%
train_labels = train_mfcc_y;
test_labels = test_mfcc_y;

cr = GMM(train_mfcc,train_labels,test_mfcc,test_labels);
