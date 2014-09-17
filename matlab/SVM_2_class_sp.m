function [train,validation_set,test_set,model] = SVM_2_class_sp(train,train_labels,validation,test,test_labels,label)

train_set = [];
test_set = [];
tr_labels = [];
te_labels = [];
pool = linspace(1,10,10);
for i = 1:length(pool)
    backup{i} = [];
end

for i = 1:length(train_labels)
    if train_labels(i) == label
        train_set = [train_set;train(i,:)];
        tr_labels = [tr_labels;train_labels(i)];
    else 
        backup{train_labels(i)} = [backup{train_labels(i)};train(i,:)];
    end
end

for i = 1:length(backup)
    if isempty(backup{i})
        continue;
    end
    len = size(backup{i},1);
    idx = randperm(len);
    idx = idx(1:11);
    train_set = [train_set;backup{i}(idx,:)];
    tr_labels = [tr_labels;zeros(11,1)];
end

    
m = mean(train_set);
v = std(train_set);
for i = 1:size(train_set,2)
   train_set(:,i) = (train_set(:,i) - m(i) ) / v(i);
end
validation_set = z_score(validation,m,v);
test_set = z_score(test,m,v);
train = z_score(train,m,v);
%
%[train_set,test_set] = scaleForSVM(train_set,test_set,0,1);
% train_set = mapminmax(train_set',0,1);
% train_set = train_set';
% 
% test_set = mapminmax(test',0,1);
% 
% test_set = test_set';
for i = 1:length(test_labels)
    if test_labels(i)~= label
        test_labels(i) = 0 ;
    end
end
model = svmtrain(tr_labels,train_set,'-b 1');
[~,a] = svmpredict(test_labels,test_set,model);

end


