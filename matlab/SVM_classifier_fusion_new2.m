function [res,correct_rate] = SVM_classifier_fusion_new2(train1,train2,train_labels,test1,test2,test_labels,blocksize,hopsize,alpha,normal)
%
if nargin < 10
    normal = 1;
end
    
%% training phase of mfcc
res = zeros(2,3);

load('lfp_blocksize_pca');
train_x = [];
train_y = [];

train_HPSS = [];

if nargin<7
    blocksize = 128;
end

if nargin < 8   
    hopsize = 32;
end

for i = 1: length(train1)
    len = size(train1{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(train1{i}(j:j+blocksize,:));
            var = std(train1{i}(j:j+blocksize,:));
            dif = diff(train1{i}(j:j+blocksize,:));
            dif2 = diff(dif);
            m_diff = mean(dif);
            m_var = std(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            com = [m,var,m_var,var_dif2];
            train_x = [train_x;com];
            train_y = [train_y;train_labels(i)];
        end
    end
end
       
for i = 1: length(train2)
    len = size(train2{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(train2{i}(j:j+blocksize,1:20));
            var = std(train2{i}(j:j+blocksize,1:20));
            dif = diff(train2{i}(j:j+blocksize,1:20));
            dif2 = diff(dif);
            m_diff = mean(dif);
            m_var = std(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            com = [m,var,m_var,var_dif2];
            train_HPSS = [train_HPSS;com];
        end
    end
end
       
        
%% normalization
if normal
     for i = 1:size(train_x,2)
         tr_maxim(i) = std(train_x(:,i));
         tr_minim(i) = mean(train_x(:,i));
         train_x(:,i) = (train_x(:,i) -tr_minim(i))/(tr_maxim(i));
     end
end
if normal
     for i = 1:size(train_HPSS,2)
         HPSS_std(i) = std(train_HPSS(:,i));
         HPSS_mean(i) = mean(train_HPSS(:,i));
         train_HPSS(:,i) = (train_HPSS(:,i) -HPSS_mean(i))/(HPSS_std(i));
     end
end


%  
SVMstruct = svmtrain(train_y,train_x,['-c 1.25, -b 1']);
SVMstruct1 = svmtrain(train_y,train_HPSS,['-c 1.25 -b 1']);
%}
%% training phase of other features
train_lfp = [];
label_lfp = [];
for i = 1:length(train_sample)
    train_lfp = [train_lfp;train_sample{i}(:,1:40)];
    label_lfp = [label_lfp;ones(size(train_sample{i}),1)*(train_labels(i))];
end

if normal 
    for i = 1:size(train_lfp,2)
	lfp_mean(i) = mean(train_lfp(:,i));
    	lfp_std(i) = std(train_lfp(:,i));
        train_lfp(:,i) = (train_lfp(:,i) - lfp_mean(i)) / lfp_std(i);
    end
%    train_lfp = mapminmax(train_lfp',0,1);
%    train_lfp = train_lfp';
end

SVMstruct2 = svmtrain(label_lfp,train_lfp,['-c 1.0, -b 1']);

%% testphase

correct = 0 ;
mfcc_correct = 0 ;
HPSS_correct = 0 ;
lfp_correct = 0 ;
mfcc_lik = 0 ;
HPSS_lik = 0;
lfp_lik = 0 ;
for i = 1:length(test1)
    test_input = [];
    test_label = [];
    test_HPSS = [];
 
    len = size(test1{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(test1{i}(j:j+blocksize,:));
            var = std(test1{i}(j:j+blocksize,:));
            dif = diff(test1{i}(j:j+blocksize,:));
            dif2 = diff(dif);

            m_diff = mean(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
            test_input = [test_input;com];
            test_label = [test_label;test_labels(i)];
        end
    end
    len = size(test2{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(test2{i}(j:j+blocksize,1:20));
            var = std(test2{i}(j:j+blocksize,1:20));
            dif = diff(test2{i}(j:j+blocksize,1:20));
            dif2 = diff(dif);

            m_diff = mean(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
            test_HPSS = [test_HPSS;com];
        end
    end
   
    test_lfp = test_sample{i}(:,1:40);
    if normal
        test_input = z_score(test_input,tr_minim,tr_maxim);
        test_lfp = z_score(test_lfp,lfp_mean,lfp_std);
      %  test_lfp = mapminmax(test_lfp',0,1);
      %  test_lfp = test_lfp';
        test_HPSS = z_score(test_HPSS,HPSS_mean,HPSS_std);
    end

    [predict1,acc,prob1] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
    [predict2,acc,prob2] = svmpredict(ones(size(test_sample{i},1),1) * (test_labels(i)),test_lfp,SVMstruct2,'-b 1');
    [predict3,acc,prob3] = svmpredict(test_label,test_HPSS,SVMstruct1,'-b 1');
    if mode(predict1) == test_labels(i)
        mfcc_correct = mfcc_correct + 1;
    end
    if mode(predict2) == test_labels(i)
        lfp_correct = lfp_correct + 1;
    end
    if mode(predict3) == test_labels(i);
        HPSS_correct = HPSS_correct + 1;
    end 
    lik1 = sum(prob1(:,test_labels(i)))/sum(max(prob1'));
    lik2 = sum(prob2(:,test_labels(i)))/sum(max(prob2'));
    lik3 = sum(prob3(:,test_labels(i)))/sum(max(prob3'));
    mfcc_lik = mfcc_lik + lik1;
    lfp_lik = lfp_lik + lik2;
    HPSS_lik = HPSS_lik + lik3;
     %{
    if length(find(predict1 == mode(predict1))) > length(predict1)/2
        predict = predict1 
    else if length(find(predict2 == mode(predict2))) > length(predict2)/2
        predict = predict2
    else    
        prob = alpha * mean(prob1) + (1-alpha) * mean(prob2); 
    
        [m,predict] = max(prob');
    end
    %}

    predict = [predict1;predict2;predict3];
    if mode(predict) == test_labels(i)
	correct = correct + 1;
    end
end
res(1,1) = mfcc_correct / length(test_labels);
res(1,2) = lfp_correct / length(test_labels);
res(1,3) = HPSS_correct / length(test_labels);
res(2,1) = mfcc_lik / length(test_labels);
res(2,2) = lfp_lik / length(test_labels);
res(2,3) = HPSS_lik / length(test_labels);

correct_rate = correct / length(test_labels);

end
