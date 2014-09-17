function [alpha,model,norm_mean,norm_std] = SVM_boosting() 




alpha = [];



load('mfcc_info');
labels = train_mfcc_y;
train_len = length(train_mfcc_x);
weight = ones(1,train_len);


[train_mfcc] = frame2song(train_mfcc_x,256,64);

mean_mfcc = mean(train_mfcc);
std_mfcc = std(train_mfcc);
norm_mean{1} = mean_mfcc;
norm_std{1} = std_mfcc;
for i = 1:size(train_mfcc,2)
    train_mfcc(:,i) = (train_mfcc(:,i) - mean_mfcc(i))/std_mfcc(i);
end


%[validation_mfcc] = frame2song(validation_mfcc_x,256,64);
%[train1,labels] = resample(train_mfcc,weight,train_len,labels);
train1 = train_mfcc;
[predict1,cr,model{1}] = SVM_class(train1,labels);
alpha = [alpha, log((cr+0.8)/(1-cr))];
weight = WeightChange(predict1,labels,weight);


load('sp_pca');
mean_sp = mean(train_sp);
std_sp = std(train_sp);
norm_mean{2} = mean_sp;
norm_std{2} = std_sp;
for i = 1:size(train_sp,2)
    train_sp(:,i) = (train_sp(:,i) - mean_sp(i))/std_sp(i);
end

[train2,labels] = resample(train_sp,weight,train_len,labels);


[predict2,cr,model{2}] = SVM_class(train2,labels);
alpha = [alpha, log((cr+0.8)/(1-cr))];
weight = WeightChange(predict2,labels,weight);



load('dsp_pca');
mean_dsp = mean(train_dsp);
std_dsp = std(train_dsp);
norm_mean{3} = mean_dsp;
norm_std{3} = std_dsp;
for i = 1:size(train_dsp,2)
    train_dsp(:,i) = (train_dsp(:,i) - mean_dsp(i))/std_dsp(i);
end

[train3,labels] = resample(train_dsp,weight,train_len,labels);

[predict3,cr,model{3}] = SVM_class(train3,labels);

alpha = [alpha, log((cr+0.8)/(1-cr))];
weight = WeightChange(predict3,labels,weight);


 
load('vdsp_pca');
mean_vdsp = mean(train_vdsp);
std_vdsp = std(train_vdsp);
norm_mean{4} = mean_vdsp;
norm_std{4} = std_vdsp;
for i = 1:size(train_vdsp,2)
    train_vdsp(:,i) = (train_vdsp(:,i) - mean_vdsp(i))/std_vdsp(i);
end

[train4,labels] = resample(train_vdsp,weight,train_len,labels);
[predict4,cr,model{4}] = SVM_class(train4,labels);
alpha = [alpha, log((cr+0.8)/(1-cr))];
weight = WeightChange(predict4,labels,weight);
%  
% load('cp_pca');
% %
% [train4,labels] = resample(train_cp,weight,train_len,labels);
% %
% [predict4,cr,model{4}] = SVM_class(train4,labels);
%alpha = [alpha, log((cr+0.8)/(1-cr))];
%weight = WeightChange(predict4,labels,weight);
%
%alpha = [alpha,0] ; 
load('scp_pca');
mean_scp = mean(train_scp);
std_scp = std(train_scp);
norm_mean{5} = mean_scp;
norm_std{5} = std_scp;
for i = 1:size(train_scp,2)
    train_scp(:,i) = (train_scp(:,i) - mean_scp(i))/std_scp(i);
end


[train5,labels] = resample(train_scp,weight,train_len,labels);
[predict5,cr,model{5}] = SVM_class(train5,labels);
alpha = [alpha, log((cr+0.8)/(1-cr))];
weight = WeightChange(predict5,labels,weight);


% load('lfp_pca');
% [train6,labels] = resample(train_lfp,weight,train_len,labels);
% [predict6,cr,model{6}] = SVM_class(train6,labels);
% %alpha = [alpha, log((cr+0.8)/(1-cr))];
% alpah = [alpha,0];





end



