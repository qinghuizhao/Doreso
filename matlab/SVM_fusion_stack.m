function [cf,correct_rate] = SVM_fusion_stack(train1,train2,train_labels,validation1,validation2,validation_labels,test1,test2,test_labels,blocksize,hopsize,normal)
%
cf = zeros(10,10);

if nargin < 12
    normal = 1;
end

%% first training phase
load('lfp_hopsize_pca');
load('GMM');

if nargin < 10
    blocksize = 128;
end
if nargin < 11  
    hopsize = 64;
end
train_x = [];
train_y = [];

for i = 1: length(train1)
    len = size(train1{i},1);
%    lfp_temp = train_sample{i};
    train_mfcc = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [train1{i}(j:j+blocksize,:),train2{i}(j:j+blocksize,1:20)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            m_diff = mean(dif);
            m_var = std(dif);
            dif2=diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            com = [m,var,m_var,var_dif2];
            train_mfcc = [train_mfcc;com];
            train_y = [train_y;train_labels(i)];
        end
    end
   % train_x = [train_x;[train_mfcc,lfp_temp(1:size(train_mfcc,1),1:40)]];
    train_x = [train_x;train_mfcc];
end
    
%% normalization
if normal == 1
     for i = 1:size(train_x,2)
         tr_maxim(i) = std(train_x(:,i));
         tr_minim(i) = mean(train_x(:,i));
         train_x(:,i) = (train_x(:,i) -tr_minim(i))/(tr_maxim(i));
     end
end

%cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -b 1'];

SVMstruct = svmtrain(train_y,train_x,['-b 1']);

%% second-level training




val_train_input = [];
val_train_label = [];

for i = 1:length(validation1)
    val_input = [];
    val_label_temp = [];

    
 %   lfp_temp = validation_sample{i};
    len = size(validation1{i},1);
    val_mfcc = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [validation1{i}(j:j+blocksize,:),validation2{i}(j:j+blocksize,1:20)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            m_diff = mean(dif);
            m_var = std(dif);
            dif2=diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            com = [m,var,m_var,var_dif2];

            val_mfcc = [val_mfcc;com];
            val_label_temp = [val_label_temp;validation_labels(i)];
        end
    end
    
  %  val_input = [val_input;[val_mfcc,lfp_temp(1:size(val_mfcc,1),1:40)]];
    val_input = [val_input;val_mfcc];
    if normal == 1
        val_input = z_score(val_input,tr_minim,tr_maxim);
    end 
    [predict,acc,prob] = svmpredict(val_label_temp,val_input,SVMstruct,'-b 1');
%    for k = 1:length(predict)
%        prob(k,:) = prob(k,:) .* 2.^(validation_data(i,:));
%    end
%     n_dim = 10;
%     nn_dim = 20;
%      val_train_temp = zeros(1,n_dim);
%      for j = 1:length(predict)
%          val_train_temp(predict(j)) = val_train_temp(predict(j)) + 1;
%      end
%    
%      val_train_input = [val_train_input;val_train_temp];
%      val_train_label = [val_train_label;val_label(i)];
    val_train_input = [val_train_input;prob];
    val_train_label = [val_train_label;val_label_temp];
end
%
%prob_mean = mean(val_train_input);
%prob_std = std(val_train_input);
%val_train_input = z_score(val_train_input,prob_mean,prob_std);
 
SVMstruct2 = svmtrain(val_train_label,val_train_input);

%}
%% testphase

correct = 0 ;
for i = 1:length(test1)
	
    
%    lfp_temp = test_sample{i};
    
    len = size(test1{i},1);
    test_input = [];
    test_label = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            temp = [test1{i}(j:j+blocksize,:),test2{i}(j:j+blocksize,1:20)];

            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            m_diff = mean(dif);
            m_var = std(dif); 
            dif2=diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            com = [m,var,m_var,var_dif2];


            test_input = [test_input;com];
        
            test_label = [test_label;test_labels(i)];
        end
    end
%    test_input = [test_input,lfp_temp(1:size(test_input,1),1:40)];
    
    if normal == 1
        test_input = z_score(test_input,tr_minim,tr_maxim);
    end
    
    [predict,~,prob] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
%    if length(find(predict==mode(predict))) > length(predict)/3 * 2
%        final_predict = predict;
%    else
%      n_dim = 10;
%%       test_input_temp = zeros(1,n_dim);
%%       for j = 1:length(predict)
%%           test_input_temp(predict(j)) = test_input_temp(predict(j)) + 1;
%%       end    
%%      final_predict = svmpredict(label(i),test_input_temp,SVMstruct2);
%      final_predict = svmpredict(ones(length(predict),1) * test_labels(i),prob,SVMstruct2);
%    end
%    prob = z_score(prob,prob_mean,prob_std);
%    for k = 1:length(predict)
%        prob(k,:) = prob(k,:) .* 2.^(test_data(i,:));
%    end
    final_predict = svmpredict(ones(length(predict),1) * test_labels(i),prob,SVMstruct2);
    if mode(final_predict) == test_labels(i)
        correct = correct + 1;
    end
    cf(mode(final_predict),test_labels(i)) =  cf(mode(final_predict),test_labels(i)) + 1;
end
correct_rate = correct / length(test_labels);

end
