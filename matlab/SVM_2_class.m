function correct_rate = SVM_2_class(train,train_labels,test,test_labels,blocksize,hopsize,normal,thres)
%
if nargin < 7
    normal = 1;
    thres = 0; 
end

%% training phase


train_x = [];
train_y = [];

if nargin < 5 
    blocksize = 128;
end
if nargin < 6
    hopsize = 32;
end

count = 1;
tr_labels = [];

for i = 1:length(train)
    if train_labels(i) == 5 || train_labels(i) == 6
        train_set{count} = train{i};
        tr_labels = [tr_labels,train_labels(i)];
        count = count + 1;
    end
end
count = 1;
te_labels = [];
for i = 1:length(test)
    if test_labels(i) == 5 || test_labels(i) == 6
        test_set{count} = test{i};
        te_labels = [te_labels,test_labels(i)];
        count = count + 1;
    end
end

for i = 1: length(train_set)
    len = size(train_set{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(train_set{i}(j:j+blocksize,:));
            var = std(train_set{i}(j:j+blocksize,:));
            dif = diff(train_set{i}(j:j+blocksize,:));
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
            train_x = [train_x;com];
            train_y = [train_y;tr_labels(i)];
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

%  
SVMstruct = svmtrain(train_y,train_x,['-c 1.0, -b 1']);
%}
%% testphase

correct = 0 ;
for i = 1:length(test_set)
    test_input = [];
    test_label = [];
    
    len = size(test_set{i},1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(test_set{i}(j:j+blocksize,:));
            var = std(test_set{i}(j:j+blocksize,:));
            dif = diff(test_set{i}(j:j+blocksize,:));
            m_diff = mean(dif);
            m_var = std(dif);
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);

            com = [m,var,m_var,var_dif2];
            test_input = [test_input;com];
            test_label = [test_label;te_labels(i)];
        end
    end
    if normal
        test_input = z_score(test_input,tr_minim,tr_maxim);
    end
	[predict,acc,prob] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
%{    
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
	if mode(predict) == te_labels(i)
		correct = correct + 1;
 	end
end


correct_rate = correct / length(test_set);
clear test_set;
end
