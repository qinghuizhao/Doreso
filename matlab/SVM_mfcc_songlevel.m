function [cf,correct_rate] = SVM_mfcc_songlevel(input,blocksize,hopsize,normal,thres)
%
if nargin < 4
    normal = 1;
    thres = 0.5 ; 
end

%% training phase

load(input);

train_x = [];
train_y = [];
cf = zeros(10,10);
if nargin < 2 
    blocksize = 128;
    hopsize = 64
end
for i = 1: length(train_mfccP_x)
    len = size(train_mfccP_x{i},1);
    temp = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(train_mfccP_x{i}(j:j+blocksize,1:13));
            var = std(train_mfccP_x{i}(j:j+blocksize,1:13));
            dif = diff(train_mfccP_x{i}(j:j+blocksize,1:13));
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
           %com = [m,var];
            temp = [temp,com];
        end
    end
    
    train_x = [train_x;temp];
     train_y = [train_y;train_mfcc_y(i)];
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

%[bestacc,bestc,bestg] =  psoSVMcgForClass(train_y,train_x);  
%cmd = ['' '-c ' num2str(bestc) , ' -g ' num2str(bestg), ' -b 1' '' ];
cmd = ['' '-c ' num2str(1.0)  ' -b 1' '' ];
SVMstruct = svmtrain(train_y,train_x,cmd);
%}
%% testphase

correct = 0 ;
for i = 1:length(test_mfccP_x)
    test_input = [];
    test_label = [];
    
    len = size(test_mfccP_x{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(test_mfccP_x{i}(j:j+blocksize,1:13));
            var = std(test_mfccP_x{i}(j:j+blocksize,1:13));
            dif = diff(test_mfccP_x{i}(j:j+blocksize,1:13));
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
         % com = [m,var];
              test_input = [test_input,com];
        end
    end
     test_label = [test_label;test_mfcc_y(i)];
    if normal
        test_input = z_score(test_input,tr_minim,tr_maxim);
    end
    
    [predict,acc,prob] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
    if mode(predict) == test_mfcc_y(i)
	correct = correct + 1;
    end
     cf(mode(predict),test_mfcc_y(i)) =  cf(mode(predict),test_mfcc_y(i)) + 1;
end


correct_rate = correct / length(test_mfccP_x);
clear test_mfccP_x;
end
