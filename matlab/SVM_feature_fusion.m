function [cf,correct_rate] = SVM_feature_fusion(train1,train2,train_labels,test1,test2,test_labels,blocksize,hopsize,normal,thres)
%

cf = zeros(10,10);
if nargin < 9
    normal = 1;
    thres = 0.1 ; 
end

%load('lfp_hopsize32_prime_pca');
%load('lfp_hopsize64_prime_pca');
%load('GMM_prime');
load('lfp_hopsize64_pca');

%% training phase

train_x = [];
train_y = [];

if nargin < 7 
    blocksize = 128;
end
if nargin < 8 
    
    hopsize = 32;
end
train_GMM  = [];
for i = 1: length(train1)
    len = size(train1{i},1);
    train_mfcc = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [train1{i}(j:j+blocksize,2:end)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            m_diff = mean(dif);
            com = [m,var,m_diff];
            train_mfcc = [train_mfcc;com];
            train_y = [train_y;train_labels(i)];
        end
    end
 %  train_x = [train_x;train_mfcc,tmp(1:size(train_mfcc,1),1:40)];
    train_GMM = [train_GMM;train_mfcc];
end
tr_m = mean(train_GMM);
tr_var = std(train_GMM);
train_block = z_score(train_GMM,tr_m,tr_var);



for i = 1:10
    train_class{i} = [];
end

for i = 1:length(train_y)
    train_class{train_y(i)} = [train_class{train_y(i)};train_block(i,:)];
end


for i = 1:10
    obj{i} = gmdistribution.fit(train_class{i},10,'CovType','diagonal');
end

train_prob = [];
for i = 1:size(train_GMM,1)
    prob = [];
    for j = 1:10
        P = posterior(obj{j},train_GMM(i,:));
        prob = [prob,P];
    end
    train_prob = [train_prob;P];
end



for i = 1: length(train1)
    len = size(train1{i},1);
    train_mfcc = [];
    if i <= length(train_sample)
       tmp = train_sample{i};
    else 
       tmp = validation_sample{i-length(train_sample)};
    end
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [train1{i}(j:j+blocksize,2:end),train2{i}(j:j+blocksize,2:20)];
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
        end
    end
   train_x = [train_x;train_mfcc,tmp(1:size(train_mfcc,1),1:40)];
 %   train_x = [train_x;train_mfcc];
end
train_x = [train_x,train_prob];

clear train1;
clear train2;       
        
%% normalization
if normal
     for i = 1:size(train_x,2)
         tr_maxim(i) = std(train_x(:,i));
         tr_minim(i) = mean(train_x(:,i));
         train_x(:,i) = (train_x(:,i) -tr_minim(i))/(tr_maxim(i));
     end
end



% [bestacc,bestc,bestg] = gaSVMcgForClass(train_y,train_x);
% 
% bestc
% bestg
% 
% save bestParaForMFCCLFP bestc bestg;


cmd = ['-c 1.25 -b 1'];

SVMstruct = svmtrain(train_y,train_x,cmd);
%}
%% testphase

correct = 0 ;
%load('sp_pca');
%[test_sp,model] = SVM_2_class_sp(train_sp,train_labels,test_sp,test_labels);
for i = 1:length(test1)
    
    len = size(test1{i},1);
    GMM_test = [];
    test_input = [];
    test_label = [];
    tmp = test_sample{i};
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [test1{i}(j:j+blocksize,2:end),test2{i}(j:j+blocksize,2:20)];
            gmm_temp = test1{i}(j:j+blocksize,2:end);
            gmm_m = mean(gmm_temp);
            gmm_var = std(gmm_temp);
            gmm_dif = diff(gmm_temp);
            gmm_m2 = mean(gmm_dif);
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);

            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
            gmm_com = [gmm_m,gmm_var,gmm_m2];
            GMM_test = [GMM_test;gmm_com];

            test_input = [test_input;com];
            test_label = [test_label;test_labels(i)];
        end
    end
    test_prob = [];
    for j = 1:10
        P = posterior(obj{j},GMM_test);
        test_prob = [test_prob,P];
    end
%    test_prob = z_score(test_prob,prob_m,prob_var);
        
    test_input = [test_input,tmp(1:size(test_input,1),1:40),test_prob];
    
    if normal
        test_input = z_score(test_input,tr_minim,tr_maxim);
    end
	[predict,acc,prob] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
    
%    ignore = [];
%    for j = 1:length(predict)
%         l = predict(j);
%         if l == 10 
%             continue;
%         end
%         [start,over] = equalsum(l);
%         if mean(prob(j,start:over))<0.4
%             ignore = [ignore,j];
%         end
%        if prob(j,predict(j)) < thres
%            ignore = [ignore,j];
%        end
%    end
%    predict(ignore) = [];
    %}
   % for j = length(predict)
   %     prob(j,:) = prob(j,:) .* (exp( test_data(i,:)));
   % end
   % [~,predict] = max(prob');
    final = mode(predict);
%     if final == 5 || final == 6
%         [p,~,prob] = svmpredict(5,test_sp(i,:),model);
%         if prob[p-4] > 0.7
%             final = p ;
%         end
%     end 

    if final == test_labels(i)
	correct = correct + 1;
    end
     cf(mode(predict),test_labels(i)) =  cf(mode(predict),test_labels(i)) + 1;     
end
correct_rate = correct / length(test_labels);

end
