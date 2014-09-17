function correct_rate = SVM_aggregate_onset(input,normal,thres)
%
if nargin < 4
    normal = 1;
    thres = 0.4 ; 
end

%% training phase

load(input);
load('Onset_smooth_info');

train_x = [];
train_y = [];



    
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
SVMstruct = svmtrain(train_y,train_x,['-c 1.0, -b 1']);
%}
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
	if mode(predict) == test_mfcc_y(i)
		correct = correct + 1;
 	end
end


correct_rate = correct / length(test_mfccP_x);
clear test_mfccP_x;
end
