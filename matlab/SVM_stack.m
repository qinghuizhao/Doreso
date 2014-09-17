function correct_rate = SVM_stack(train_path,validation_path,test_path,blocksize,hopsize,normal)

if nargin < 6
    normal = 1;
end

%% first training phase

train_label = [train_path,'tz_label.txt'];

filelist = getAllFiles([train_path,'train/']);

train_x = [];
train_y = [];

if nargin<3 
    blocksize = 100;
    hopsize = 50;
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
if normal == 1
     for i = 1:size(train_x,2)
         tr_maxim(i) = std(train_x(:,i));
         tr_minim(i) = mean(train_x(:,i));
         train_x(:,i) = (train_x(:,i) -tr_minim(i))/(tr_maxim(i));
     end
end
%  
SVMstruct = svmtrain(train_y,train_x,['-c 1.0, -b 1']);

%% second-level training
val_label = [validation_path,'tz_label.txt'];
val_label = readfile(val_label);

validation_name = [validation_path,'validation/'] ;

valfiles = getAllFiles(validation_name);

val_train_input = [];
val_train_label = [];

for i = 1:length(valfiles)
    val_input = [];
    val_label_temp = [];
	temp_val = readfile(valfiles{i});
    
    len = size(temp_val,1);
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(temp_val(j:j+blocksize,:));
            var = std(temp_val(j:j+blocksize,:));
            com = [m,var];
            val_input = [val_input;com];
            val_label_temp = [val_label_temp;val_label(i)];
        end
    end
	if normal == 1
        val_input = z_score(val_input,tr_minim,tr_maxim);
    end 
	[predict,acc,prob] = svmpredict(val_label_temp,val_input,SVMstruct,'-b 1');
    
    n_dim = 10;
    %nn_dim = 20;
%     val_train_temp = zeros(1,n_dim);
%     for j = 1:length(predict)
%         val_train_temp(predict(j)) = val_train_temp(predict(j)) + 1;
%     end
    
%     val_train_input = [val_train_input;val_train_temp];
%     val_train_label = [val_train_label;val_label(i)];
     val_train_input = [val_train_input;prob];
     val_train_label = [val_train_label;ones(length(predict),1)*val_label(i)];
end

SVMstruct2 = svmtrain(val_train_label,val_train_input,'-b 1`');

%}
%% testphase
labelname = [test_path,'tz_label.txt'];
label = readfile(labelname);
%}
testname = [test_path,'test/'];

testfiles = getAllFiles(testname);

correct = 0 ;
for i = 1:length(testfiles)
	temp_test = readfile(testfiles{i});
    
    len = size(temp_test,1);
    test_input = [];
    test_label = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(temp_test(j:j+blocksize,:));
            var = std(temp_test(j:j+blocksize,:));
            com = [m,var];
            test_input = [test_input;com];
            test_label = [test_label;label(i)];
        end
    end
    if normal == 1
        test_input = z_score(test_input,tr_minim,tr_maxim);
    end
	[predict,acc,prob] = svmpredict(test_label,test_input,SVMstruct,'-b 1');
    %
%     ignore = [];
%     for j = 1:length(predict)
%         l = predict(j);
%         if l == 10 
%             continue;
%         end
%         [start,over] = equalsum(l);
%         if mean(prob(j,start:over))<0.4
%             ignore = [ignore,j];
%         end
%     end
%     predict(ignore) = [];
%     %}
    n_dim = 10;
%     test_input_temp = zeros(1,n_dim);
%     for j = 1:length(predict)
%         test_input_temp(predict(j)) = test_input_temp(predict(j)) + 1;
%     end    
%     final_predict = svmpredict(label(i),predict(1:n_dim),SVMstruct2);
    final_predict = svmpredict(ones(length(predict),1) * label(i),prob,SVMstruct2);
	if mode(final_predict) == label(i)
		correct = correct + 1;
 	end
end

correct_rate = correct / length(testfiles);

end
