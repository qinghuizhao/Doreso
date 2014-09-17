addpath('../data/');

load('mfcc_info');
load('Pitch_info');

for i = 1:length(Pitch_train)
    mfccP_train_x{i} = [train_mfcc_x{i}(:,2:end),Pitch_train{i}'];
end

for i = 1:length(Pitch_test)
    mfccP_test_x{i} = [test_mfcc_x{i}(:,2:end),Pitch_test{i}'];
end


for i = 1:length(Pitch_validation)
    mfccP_validation_x{i} = [validation_mfcc_x{i}(:,2:end),Pitch_validation{i}'];
end
