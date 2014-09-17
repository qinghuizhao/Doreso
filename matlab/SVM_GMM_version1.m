function [cf,cr] = SVM_GMM_version1(trainfile,testfile,train_labels,test_labels)

cf = zeros(10,10);
train_data = readfile(trainfile);



test_data = readfile(testfile);

%train_data = [train_data;validation_data];
%train_labels = [train_labels;validation_labels];


m = mean(train_data);
s = std(train_data);
for i = 1:size(train_data,2)
    train_data(:,i) = exp(train_data(:,i));
    train_data(:,i) = (train_data(:,i)-m(i))/s(i);
end

for i = 1:size(test_data,2)
    test_data(:,i) = exp(test_data(:,i));
end

test_data = z_score(test_data,m,s);
%train_data = mapminmax(train_data',0,1);
%train_data = train_data';
%test_data = mapminmax(test_data',0,1);
%test_data = test_data';
%

model = svmtrain(train_labels,train_data);

correct = 0 ;

for i = 1:length(test_labels)
    [p,a,prob] = svmpredict(test_labels(i),test_data(i,:),model);
    if p == test_labels(i)
        correct = correct + 1 ;
    end
    cf(p,test_labels(i)) = cf(p,test_labels(i)) + 1;

end


cr = correct / length(test_labels);

end
