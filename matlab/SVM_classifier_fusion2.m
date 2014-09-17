function correct_rate = SVM_classifier_fusion2(train_path,test_path,blocksize,hopsize,alpha,normal,thres)
%
if nargin < 6
    normal = 1;
    thres = 0.5 ; 
end
    
%% training phase of mfcc

load('lfp_blocksize_pca');

train_label = [train_path,'tz_label.txt'];

filelist = getAllFiles([train_path,'train/']);

train_x = [];
train_y = [];

if nargin<3
    blocksize = 128;
    hopsize = 64;
end
train_label = readfile(train_label);

for i = 1: length(filelist)
    filelines = readfile(filelist{i});
    len = size(filelines,1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(filelines(j:j+blocksize,:));
            var = std(filelines(j:j+blocksize,:));
            com = [m,var];
            train_x = [train_x;com];
            train_y = [train_y;train_label(i)];
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
labelname = [test_path,'tz_label.txt'];
label = readfile(labelname);
%}
testname = [test_path,'test/'];

testfiles = getAllFiles(testname);

correct = 0 ;
for i = 1:length(testfiles)
    test_input = [];
    test_label = [];
	temp_test = readfile(testfiles{i});
    
    len = size(temp_test,1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(temp_test(j:j+blocksize,:));
            var = std(temp_test(j:j+blocksize,:));
            com = [m,var];
            test_input = [test_input;com];
            test_label = [test_label;label(i)];
        end
    end
    test_lfp = test_sample{i};
    if normal
        test_input = z_score(test_input,tr_minim,tr_maxim);
        test_lfp = z_score(test_lfp,lfp_mean,lfp_std);
    end
    [predict,acc,prob1] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
    [predict2,acc,prob2] = svmpredict(ones(size(test_sample{i}),1) * (test_data_labels(i)+1),test_lfp,SVMstruct2,'-b 1');
    
    prob1 = mean(prob1);
    prob2 = mean(prob2);
    
    [m1,predict1] = max(prob1);
    [m2,predict2] = max(prob2);

    if m1 > alpha * m2
        predict = predict1;
    else
        predict = predict2;
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
	if mode(predict) == label(i)
		correct = correct + 1;
 	end
end

correct_rate = correct / length(testfiles);


