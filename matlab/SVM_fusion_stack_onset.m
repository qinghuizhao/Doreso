function correct_rate = SVM_fusion_stack_onset(input,blocksize,hopsize,normal)
%
if nargin < 4 
    normal = 1;
end

load(input);

%% first training phase
load('lfp_blocksize_pca');
load('Onset_smooth_info');

if nargin < 2 
    blocksize = 128;
    hopsize = 64;
end
train_x = [];
train_y = [];
for i = 1: length(train_mfccP_x)
    len = size(train_mfccP_x{i},1);
    lfp_temp = train_sample{i};
    train_mfcc = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(train_mfccP_x{i}(j:j+blocksize,:));
            var = std(train_mfccP_x{i}(j:j+blocksize,:));
            dif = diff(train_mfccP_x{i}(j:j+blocksize,:));
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_diff,m_var];
            com  = [m,var];
            train_mfcc = [train_mfcc;com];
            train_y = [train_y;train_mfcc_y(i)];
        end
    end
    train_x = [train_x;[train_mfcc,lfp_temp(:,1:40)]];
end
clear train_mfccP_x;
    
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

for i = 1:length(validation_mfccP_x)
    val_input = [];
    val_label_temp = [];

    
    lfp_temp = validation_sample{i};
    len = size(validation_mfccP_x{i},1);
    val_mfcc = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(validation_mfccP_x{i}(j:j+blocksize,:));
            var = std(validation_mfccP_x{i}(j:j+blocksize,:));
            dif = diff(validation_mfccP_x{i}(j:j+blocksize,:));
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_diff,m_var];
            com = [m,var];
            val_mfcc = [val_mfcc;com];
            val_label_temp = [val_label_temp;validation_mfcc_y(i)];
        end
    end
    
    val_input = [val_input;[val_mfcc,lfp_temp(:,1:40)]];
    if normal == 1
        val_input = z_score(val_input,tr_minim,tr_maxim);
    end 
    [predict,acc,prob] = svmpredict(val_label_temp,val_input,SVMstruct,'-b 1');
    
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
clear validation_mfccP_x;
SVMstruct2 = svmtrain(val_train_label,val_train_input,'-b 1`');

%}
%% testphase

correct = 0 ;
for i = 1:length(test_mfccP_x)
	
    
    lfp_temp = test_sample{i};
    
    len = size(test_mfccP_x{i},1);
    test_input = [];
    test_label = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(test_mfccP_x{i}(j:j+blocksize,:));
            var = std(test_mfccP_x{i}(j:j+blocksize,:));
            dif = diff(test_mfccP_x{i}(j:j+blocksize,:));
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_diff,m_var];
            com = [m,var];
            test_input = [test_input;com];
        
            test_label = [test_label;test_mfcc_y(i)];
        end
    end
    test_input = [test_input,lfp_temp(:,1:40)];
    
    if normal == 1
        test_input = z_score(test_input,tr_minim,tr_maxim);
    end
    
    [predict,acc,prob] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
    if length(find(predict==mode(predict))) > length(predict)/3 * 2
        final_predict = predict;
    else
      n_dim = 10;
%       test_input_temp = zeros(1,n_dim);
%       for j = 1:length(predict)
%           test_input_temp(predict(j)) = test_input_temp(predict(j)) + 1;
%       end    
%      final_predict = svmpredict(label(i),test_input_temp,SVMstruct2);
      final_predict = svmpredict(ones(length(predict),1) * test_mfcc_y(i),prob,SVMstruct2);
    end
    if mode(final_predict) == test_mfcc_y(i)
		correct = correct + 1;
    end
end
clear test_mfccP_x;
correct_rate = correct / length(test_mfcc_y);

end
