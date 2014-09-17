function correct_rate = SVM_classifier_fusion_onset(input,blocksize,hopsize,alpha,normal)
%
if nargin < 5
    normal = 1;
    alpha = 0.8;
end
    
%% training phase of mfcc

load('lfp_winsize_pca');
load('Onset_smooth_info');

load(input);
train_x = [];
train_y = [];

if nargin<2
    blocksize = 128;
    hopsize = 64;
end
        
for i = 1: length(train_mfccP_x)
    train_frame_idx = time2frame(Onset_train_smooth{i},22050,512);
    len = size(train_mfccP_x{i},1);
    for j = 1:length(train_frame_idx)-1
        if train_frame_idx(j+1)>len
            break;
        end 
        if train_frame_idx(j+1) - train_frame_idx(j) < 10
            continue;
        end 
        block = train_mfccP_x{i}(train_frame_idx(j):train_frame_idx(j+1),:);
        m = mean(block);
        var = std(block);
        dif = diff(block);
        m_diff = mean(dif);
        m_var = std(dif);
        com = [m,var,m_diff,m_var];
        if length(com)~=52
            w = 1;
        end 
        train_x = [train_x;com];
        train_y = [train_y;train_mfcc_y(i)];
    end 
end
clear train_mfccP_x;

      
        
%% normalization
if normal
     for i = 1:size(train_x,2)
         tr_maxim(i) = std(train_x(:,i));
         tr_minim(i) = mean(train_x(:,i));
         train_x(:,i) = (train_x(:,i) -tr_minim(i))/(tr_maxim(i));
     end
end

%  
SVMstruct = svmtrain(train_y,train_x,['-c 1.25, -b 1']);
%}
%% training phase of other features
train_lfp = [];
label_lfp = [];
for i = 1:length(train_sample)
	train_lfp = [train_lfp;train_sample{i}];
    label_lfp = [label_lfp;ones(size(train_sample{i}),1)*(train_data_labels(i)+1)];
end

if normal 
	for i = 1:size(train_lfp,2)
		lfp_mean(i) = mean(train_lfp(:,i));
		lfp_std(i) = std(train_lfp(:,i));
        train_lfp(:,i) = (train_lfp(:,i) - lfp_mean(i)) / lfp_std(i);
	end
end
SVMstruct2 = svmtrain(label_lfp,train_lfp,['-c 1.0, -b 1']);

%% testphase

correct = 0 ;
for i = 1:length(test_mfccP_x)
    test_input = [];
    test_label = [];
    test_frame_idx = time2frame(Onset_test_smooth{i},22050,512);
    len = size(test_mfccP_x{i},1);
    for j = 1:length(test_frame_idx)-1
        if test_frame_idx(j+1)>len
            break;
        end
        block = test_mfccP_x{i}(test_frame_idx(j):test_frame_idx(j+1),:);
        m = mean(block);
        var = std(block);
        dif = diff(block);
        m_diff = mean(dif);
        m_var = std(dif);
        com = [m,var,m_diff,m_var];
        test_input = [test_input;com];
        test_label = [test_label;test_mfcc_y(i)];
    end
    test_lfp = test_sample{i};
    if normal
        test_input = z_score(test_input,tr_minim,tr_maxim);
        test_lfp = z_score(test_lfp,lfp_mean,lfp_std);
    end
    [predict1,acc,prob1] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
    [predict2,acc,prob2] = svmpredict(ones(size(test_sample{i}),1) * (test_data_labels(i)+1),test_lfp,SVMstruct2,'-b 1');
    if length(find(predict1 == mode(predict1))) > length(predict1)/2
        predict = predict1 
    else if length(find(predict2 == mode(predict2))) > length(predict2)/2
        predict = predict2
    else    
        prob = alpha * mean(prob1) + (1-alpha) * mean(prob2); 
    
        [m,predict] = max(prob');
    end
%     ignore = [];
%     for j = 1:length(predict)
% %         l = predict(j);
% %         if l == 10 
% %             continue;
% %         end
% %         [start,over] = equalsum(l);
% %         if mean(prob(j,start:over))<0.4
% %             ignore = [ignore,j];
% %         end
%         if prob(j,predict(j)) < thres
%             ignore = [ignore,j];
%         end
%     end
%     predict(ignore) = [];
%     %}
    if mode(predict) == test_mfcc_y(i)
    	correct = correct + 1;
    end
end

correct_rate = correct / length(test_mfcc_y);

end
