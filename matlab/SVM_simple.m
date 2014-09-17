function cr = SVM_simple(train_data,train_labels,test_data,test_labels)

m = mean(train_data);
var = std(train_data);

train_data = z_score(train_data,m,var);



model = svmtrain(train_labels,train_data);

correct = 0 ;

for i = 1:length(test_data)
    test_data{i} = z_score(test_data{i},m,var);
    len = size(test_data{i},1);
    [predict] = svmpredict(ones(len,1)*test_labels(i),test_data{i},model);
    if mode(predict) == test_labels(i);
        correct = correct + 1;
    end
end

cr = correct / length(test_data);

end 

