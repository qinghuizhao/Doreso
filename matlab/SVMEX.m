%% training phase
%
path = '~/tiger/data/splited/Tz/mfcc/trainset_splited/';
train_label = [path,'tz_mfcc_label.txt'];

filelist = getAllFiles([path,'train/']);


% train_x_file = fopen([path,'tz_mfcc_train.txt'],'r');
% train_y_file = fopen([path,'tz_mfcc_trlab.txt'],'r');
% test_x_file = fopen([path,'tz_max_test.txt'],'r');
% test_y_file = fopen([path,'tz_max_telab.txt'],'r');

train_x = [];
train_y = [];
test_x = [];
test_y = [];

blocksize = 100;
hopsize = 50;
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

       
            
        
% 
% while ~feof(train_x_file)
%       fileline = fgetl(train_x_file);
%       fileline = str2num(fileline);
%       train_x = [train_x ; fileline]; 
% end
% 
% fclose(train_x_file);
% count = 0 ; 
% temp = [];
% while ~feof(train_y_file)
%     fileline = fgetl(train_y_file);
%     fileline = str2num(fileline)+1;
%     train_y = [train_y;fileline];
% end
% 
% fclose(train_y_file);


% 
% while ~feof(test_x_file)
%    fileline = fgetl(test_x_file);
%    fileline = str2num(fileline);
%    test_x = [test_x;fileline];
% end
% 
% fclose(test_x_file);
% 
% 
% while ~feof(test_y_file)
%    fileline = fgetl(test_y_file);
%    fileline = str2num(fileline)+1;
%    test_y = [test_y;fileline];
% end
% 
% fclose(test_y_file);

%% normalization

 for i = 1:size(train_x,2)
     tr_maxim(i) = std(train_x(:,i));
     tr_minim(i) = mean(train_x(:,i));
     train_x(:,i) = (train_x(:,i) -tr_minim(i))/(tr_maxim(i));
 end

%  
SVMstruct = svmtrain(train_y,train_x,['-c 1.0']);
%}
%% testphase
testpath = '/media/tiger/data/splited/Tz/mfcc/testset_splited/';
labelname = [testpath,'tz_mfcc_label.txt'];
label = readfile(labelname);

testname = [testpath,'test/'];

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
	test_input = z_score(test_input,tr_minim,tr_maxim);
	
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
%{
Predict = svmpredict(test_y,test_x,SVMstruct);



size(train_x)

for i = 1:length(Predict)
    if Predict(i) == test_y(i)
        correct = correct + 1 ;
    end
end
%}
correct_rate = correct / length(testfiles)
