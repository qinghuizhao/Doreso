function [cf,correct_rate] = SVM_feature_fusion_osnet(train1,train2,train_labels,test1,test2,test_labels,blocksize,hopsize,normal,thres)
%
cf = zeros(10,10);

if nargin < 9
    normal = 1;
    thres = 0.1 ; 
end

%% training phase
load('Onset_smooth_info');
train_x = [];
train_y = [];

if nargin < 7 
    blocksize = 128;
end
if nargin < 8
    hopsize = 32;
end
for i = 1: length(train1)
    train_frame_idx = time2frame(Onset_train_smooth{i},22050,512);
    len = size(train1{i},1);
    for j = 1:length(train_frame_idx) - 1
        if train_frame_idx(j+1) > len
            break;
        end
        if train_frame_idx(j+1) - train_frame_idx(j) < 5
            continue;
        end
        temp = [train1{i}(train_frame_idx(j):train_frame_idx(j+1),:),train2{i}(train_frame_idx(j):train_frame_idx(j+1),1:20)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            dif2 = diff(dif);
            m_diff = mean(dif);
            m_var = std(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            com = [m,var,m_var,var_dif2];
        if length(com) ~= 160
            w = 1;
        end
            train_x = [train_x;com];
            train_y = [train_y;train_labels(i)];
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



% [bestacc,bestc,bestg] = gaSVMcgForClass(train_y,train_x);
% 
% bestc
% bestg
% 
% save bestParaForMFCCLFP bestc bestg;


cmd = ['-c 1.0 -b 1'];

SVMstruct = svmtrain(train_y,train_x,cmd);
%}
%% testphase

correct = 0 ;
for i = 1:length(test1)
    test_frame_idx = time2frame(Onset_test_smooth{i},22050,512); 
    len = size(test1{i},1);
    test_input = [];
    test_label = [];
    for j = 1:length(test_frame_idx) - 1
        if test_frame_idx(j+1) > len
            break;
        end
        if test_frame_idx(j+1) - test_frame_idx(j) < 5
            continue;
        end
       temp = [test1{i}(test_frame_idx(j):test_frame_idx(j+1),:),test2{i}(test_frame_idx(j):test_frame_idx(j+1),1:20)]; 
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            dif2 = diff(dif);
            m_diff = mean(dif);
            m_var = std(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            com = [m,var,m_var,var_dif2];
            test_input = [test_input;com];
            test_label = [test_label;test_labels(i)];
    end
    
    if normal
        test_input = z_score(test_input,tr_minim,tr_maxim);
    end
	[predict,acc,prob] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
    
    ignore = [];
    for j = 1:length(predict)
%         l = predict(j);
%         if l == 10 
%             continue;
%         end
%         [start,over] = equalsum(l);
%         if mean(prob(j,start:over))<0.4
%             ignore = [ignore,j];
%         end
        if prob(j,predict(j)) < thres
            ignore = [ignore,j];
        end
    end
    predict(ignore) = [];
    %}
	if mode(predict) == test_labels(i)
		correct = correct + 1;
 	end
     cf(mode(predict),test_labels(i)) =  cf(mode(predict),test_labels(i)) + 1;
end
correct_rate = correct / length(test_labels);
end
