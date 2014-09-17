function cr = GMM(train,train_labels,test,test_labels)

train_block = [];
train_y = [];

blocksize = 64;
hopsize = 32;

for i = 1: length(train)
    len = size(train{i},1);
    train_mfcc = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [train{i}(j:j+blocksize,2:end)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var];
            train_mfcc = [train_mfcc;com];
            train_y = [train_y;train_labels(i)];
        end
    end
 %  train_x = [train_x;train_mfcc,tmp(1:size(train_mfcc,1),1:40)];
    train_block = [train_block;train_mfcc];
end
tr_m = mean(train_block);
tr_var = std(train_block);
train_block = z_score(train_block,tr_m,tr_var);


for i = 1:10
    train_class{i} = [];
end

for i = 1:length(train_y)
    train_class{train_y(i)} = [train_class{train_y(i)};train_block(i,:)];
end


for i = 1:10
    obj{i} = gmdistribution.fit(train_class{i},32,'CovType','diagonal');
end

correct = 0 ;
for i = 1:length(test)
   len = size(test{i},1);
   test_mfcc = [];
   test_y = [];
   for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [test{i}(j:j+blocksize,2:end)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var];
            test_mfcc = [test_mfcc;com];
            test_y = [test_y;test_labels(i)];
        end
    end
    test_mfcc = z_score(test_mfcc,tr_m,tr_var);

    for j = 1:10
        P = posterior(obj{j},test_mfcc);
        P = sort(P,2);
        P = P(:,1:10);
        y = pdf(obj{j},test_mfcc);
        Prob(j) = sum(y);
    end
    [m,predict] = max(Prob);
    if predict == test_labels(i)
        correct = correct + 1;
    end
end

cr = correct / length(test);

end
