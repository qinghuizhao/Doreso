function correct_rate = SVM_aggregate(input,blocksize,hopsize,normal,thres)
%
if nargin < 4
    normal = 1;
    thres = 0.5 ; 
end

%% training phase

load(input);

train_x = [];
train_y = [];

if nargin < 2 
    blocksize = 100;
    hopsize = 50;
end

for i = 1: length(train_mfcc_x)
    len = size(train_mfcc_x{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(train_mfcc_x{i}(j:j+blocksize,:));
            var = std(train_mfcc_x{i}(j:j+blocksize,:));
            dif = diff(train_mfcc_x{i}(j:j+blocksize,:));
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_diff,m_var];
            train_x = [train_x;com];
            train_y = [train_y;train_mfcc_y(i)];
        end
    end
end
clear train_mfcc_x;
       
        
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
for i = 1:length(test_mfcc_x)
    test_input = [];
    test_label = [];
    
    len = size(test_mfcc_x{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(test_mfcc_x{i}(j:j+blocksize,:));
            var = std(test_mfcc_x{i}(j:j+blocksize,:));
            dif = diff(test_mfcc_x{i}(j:j+blocksize,:));
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_diff,m_var];
            test_input = [test_input;com];
            test_label = [test_label;test_mfcc_y(i)];
        end
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


correct_rate = correct / length(test_mfcc_x);
clear test_mfcc_x;
end
