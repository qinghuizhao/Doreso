function [new_feature,new_acc,acc] = FFS(trainset1,trainset2,train_labels,testset1,testset2,test_labels)

blocksize = 128;
hopsize = 64;


load('lfp_hopsize64_pca');
load('selectFeature');

train_x = [];
train_y = [];

for i = 1:length(trainset1)
    len = length(trainset1{i});
    train_mfcc = [];
    tmp = train_sample{i};
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [trainset1{i}(j:j+blocksize,1:end),trainset2{i}(j:j+blocksize,1:end)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
            train_mfcc = [train_mfcc;com];
            train_y = [train_y;train_labels(i)];
        end
    end
    train_mfcc = train_mfcc(:,new_feature);
   % train_x = [train_x;train_mfcc];
    train_x = [train_x;train_mfcc,tmp(1:size(train_mfcc,1),1:40)];
end  
[n_sample,dimen] = size(train_x);

%tr_m = mean(train_x);
%tr_var = std(train_x);
%train_x = z_score(train_x,tr_m,tr_var);

for i = 1:length(testset1)
    len = size(testset1{i},1) ;
    tmp = test_sample{i};
    test_input = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [testset1{i}(j:j+blocksize,1:end),testset2{i}(j:j+blocksize,:)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);

            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
            test_input = [test_input;com];
        end
    end
    test_input = test_input(:,new_feature);
    test_input = [test_input,tmp(1:size(test_input,1),1:40)];
    testset{i} = test_input;
end
l = length(new_feature);

pre_acc = 0 ;
acc = [];
new_acc = 0 ;
new_features = [];
selected_feature = [];
while length(new_features) < dimen
    tmp_acc = 0;
    for i = 1:dimen
        if isempty(find(new_features == i))
            temp_train = [train_x(:,new_features),train_x(:,i)];
            temp_new = [new_features,i];
            temp_test = selectTestFea(testset,temp_new);
            cr = SVM_simple(temp_train,train_y,temp_test,test_labels);
            if cr > tmp_acc
                tmp_selected = i;
                tmp_acc = cr;
            end
        end
    end
    pre_acc = new_acc;
    new_acc = tmp_acc;
    acc = [acc,new_acc];
    disp(tmp_selected);
    disp(new_acc);
    new_features = [new_features,tmp_selected];
end

end
