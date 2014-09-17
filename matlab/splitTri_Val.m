function [trainset,trainset2,valset,valset2,train_labels,val_labels] = splitTri_Val(data,data2,labels)

idx = randperm(length(data));
len = length(idx);
trainset = data(idx(1:round(len/2)));

valset = data(idx(round(len/2)+1:end));
trainset2 = data2(idx(1:round(len/2)));

valset2 = data2(idx(round(len/2)+1:end));

if nargin>1
val_labels = labels(idx(round(len/2)+1:end));
train_labels = labels(idx(1:round(len/2)));
end
end
