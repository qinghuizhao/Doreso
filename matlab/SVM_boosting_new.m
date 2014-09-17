function [alpha,model,norm_mean,norm_std] = SVM_boosting_new(trainset,labels,mfcc) 

alpha = [];


train_len = length(trainset);
weight = ones(1,train_len);

if mfcc
    [train_mfcc] = frame2song(trainset,256,64);
else
    train_mfcc = trainset;
end
    

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
alpha = [alpha, log((cr-0.000001)/(1-cr))];
weight = WeightChange(predict1,labels,weight);



[train2,labels] = resample(train1,weight,train_len,labels);


[predict2,cr,model{2}] = SVM_class(train2,labels);
alpha = [alpha, log((cr-0.000001)/(1-cr))];
weight = WeightChange(predict2,labels,weight);


[train3,labels] = resample(train2,weight,train_len,labels);

[predict3,cr,model{3}] = SVM_class(train3,labels);

alpha = [alpha, log((cr-0.000001)/(1-cr))];
weight = WeightChange(predict3,labels,weight);


 

[train4,labels] = resample(train3,weight,train_len,labels);
[predict4,cr,model{4}] = SVM_class(train4,labels);
alpha = [alpha, log((cr-0.000001)/(1-cr))];
weight = WeightChange(predict4,labels,weight);
%  
% load('cp_pca');
% %
% [train4,labels] = resample(train_cp,weight,train_len,labels);
% %
% [predict4,cr,model{4}] = SVM_class(train4,labels);
%alpha = [alpha, log((cr-0.000001)/(1-cr))];
%weight = WeightChange(predict4,labels,weight);
%
%alpha = [alpha,0] ; 

[train5,labels] = resample(train4,weight,train_len,labels);
[predict5,cr,model{5}] = SVM_class(train5,labels);
alpha = [alpha, log((cr-0.000001)/(1-cr))];
weight = WeightChange(predict5,labels,weight);


% load('lfp_pca');
% [train6,labels] = resample(train_lfp,weight,train_len,labels);
% [predict6,cr,model{6}] = SVM_class(train6,labels);
% %alpha = [alpha, log((cr-0.000001)/(1-cr))];
% alpah = [alpha,0];





end



