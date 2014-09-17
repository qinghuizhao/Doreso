function [ptrain_label,train_acc,model] = SVM_class(train_final,train_data_labels)


model = svmtrain(train_data_labels, train_final , '-b 1');

[ptrain_label, train_accuracy] = svmpredict(train_data_labels, train_final, model);
train_acc = train_accuracy(1)/100;
end

