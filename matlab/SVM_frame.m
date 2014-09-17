function correct_rate = SVM_frame(train_path,test_path)

%% training phase
train_label = [train_path,'tz_label.txt'];

filelist = getAllFiles([train_path,'train/']);

train_x = [];
train_y = [];

train_label = readfile(train_label);

for i = 1: length(filelist)
    filelines = readfile(filelist{i});
    filelines = subsample(filelines,100);
    train_x = [train_x;filelines];
    train_y = [train_y;ones(length(filelines),1)*train_label(i)];
end

disp('loading done');

%% normalization

 for i = 1:size(train_x,2)
     tr_maxim(i) = std(train_x(:,i));
     tr_minim(i) = mean(train_x(:,i));
     train_x(:,i) = (train_x(:,i) -tr_minim(i))/(tr_maxim(i));
 end
 
SVMstruct = svmtrain(train_y,train_x,['-c 1.25']);

disp('training done');
%}

%% testphase
labelname = [test_path,'tz_label.txt'];
label = readfile(labelname);

testname = [test_path,'test/'];

testfiles = getAllFiles(testname);

correct = 0 ;
for i = 1:length(testfiles)
	test_input = readfile(testfiles{i});
	test_input = z_score(test_input,tr_minim,tr_maxim);
	test_label = ones(size(test_input,1),1) * label(i);
	[predict,acc,prob] = svmpredict(test_label,test_input,SVMstruct);
    %
    ignore = [];
    for j = 1:length(predict)
        l = predict(j);
        if l == 10 
            continue;
        end
        [start,over] = equalsum(l);
        if mean(prob(j,start:over))<0.4
            ignore = [ignore,j];
        end
    end
    predict(ignore) = [];
    %}
	if mode(predict) == label(i)
		correct = correct + 1;
 	end
end

correct_rate = correct / length(testfiles);
end
