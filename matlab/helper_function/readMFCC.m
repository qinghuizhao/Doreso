train_path = '~/tiger/data/FeatureTxt/Prime/HPSS_MFCC/trainset_splited/' ; 
test_path =  '~/tiger/data/FeatureTxt/Prime/HPSS_MFCC/testset_splited/' ;
%validation_path = '~/tiger/data/FeatureTxt/Prime/mfcc/validationset_splited/';

train_label = [train_path,'label.txt'];

filelist = getAllFiles([train_path,'train/']);

train_mfcc_y = readfile(train_label);

for i = 1: length(filelist)
    filelines = readfile(filelist{i});
    train_mfcc_x{i} = filelines;
end
%
%%%
%val_label = [validation_path,'label.txt'];
%validation_mfcc_y = readfile(val_label);
%
%validation_name = [validation_path,'validation/'] ;
%
%valfiles = getAllFiles(validation_name);
%
%for i = 1:length(valfiles)
%	temp_val = readfile(valfiles{i});
%    validation_mfcc_x{i} = temp_val;
%end

%% testphase
labelname = [test_path,'label.txt'];
test_mfcc_y = readfile(labelname);
%}
testname = [test_path,'test/'];

testfiles = getAllFiles(testname);


for i = 1:length(testfiles)
   
	temp_test = readfile(testfiles{i});
    
    test_mfcc_x{i} = temp_test;  
end
save ../data/mfcc_prime_hpss_info train_mfcc_x train_mfcc_y  ...
    test_mfcc_x test_mfcc_y;
