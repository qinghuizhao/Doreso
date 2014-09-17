function [cf,cr] = SVM_GMM_version2(trainfile,validationfile,testfile,train_labels,validation_labels,test_labels)

cf = zeros(10,10);
%load('bar_dsp');
%load('blues_vdsp');
%load('class_vdsp');
%load('cntry_dsp');
%load('elec_dsp');
%load('jazz_sp');
%load('metal_sp');
%load('rap_sp');
%load('rock_scp');
%load('rom_sp');

%load('svm_model');
load('Pandora_scp');
load('dataset_coef');


%[train_data,test_data] = scaleForSVM(train_data,test_data);


% train_data = mapminmax(train_data',0,1);
% train_data = train_data';
% test_data = mapminmax(test_data',0,1);
% test_data = test_data';
% validation = mapminmax(validation',0,1);
% validation = validation';

train_rock_scp = train_data * rock_scp_coef;
test_rock_scp = test_data * rock_scp_coef;
validation_rock_scp = validation * rock_scp_coef;

train_rock_scp = [train_rock_scp;validation_rock_scp];

[train_rock_scp,validation_rock_sp, test_rock_scp,rock_scp] = SVM_2_class_sp(train_rock_scp,train_labels,validation_rock_scp,test_rock_scp,test_labels,9);


load('Pandora_sp');
% train_data = mapminmax(train_data',0,1);
% train_data = train_data';
% test_data = mapminmax(test_data',0,1);
% test_data = test_data';
% validation = mapminmax(validation',0,1);
% validation = validation';
%[train_data,test_data] = scaleForSVM(train_data,test_data);


train_jazz_sp = train_data * jazz_sp_coef;
test_jazz_sp = test_data * jazz_sp_coef;
validation_jazz_sp = validation * jazz_sp_coef;

[train_jazz_sp, validation_jazz_sp,test_jazz_sp,jazz_sp] = SVM_2_class_sp(train_jazz_sp,train_labels,validation_jazz_sp,test_jazz_sp,test_labels,7);

train_metal_sp = train_data * metal_sp_coef;
test_metal_sp = test_data * metal_sp_coef;
validation_metal_sp = validation * metal_sp_coef;

[train_metal_sp, validation_metal_sp, test_metal_sp,metal_sp] = SVM_2_class_sp(train_metal_sp,train_labels,validation_metal_sp,test_metal_sp,test_labels,8);

train_hip_sp = train_data * rap_sp_coef;
test_hip_sp = test_data * rap_sp_coef;
validation_hip_sp = validation * rap_sp_coef;

[train_hip_sp, validation_hip_sp, test_hip_sp,hip_sp] = SVM_2_class_sp(train_hip_sp,train_labels,validation_hip_sp, test_hip_sp,test_labels, 6);

train_rom_sp = train_data * rom_sp_coef;
test_rom_sp = test_data * rom_sp_coef;
validation_rom_sp = validation * rom_sp_coef;

[train_rom_sp, validation_rom_sp, test_rom_sp,rom_sp] = SVM_2_class_sp(train_rom_sp,train_labels,validation_rom_sp, test_rom_sp,test_labels,10);

load('Pandora_dsp');
% train_data = mapminmax(train_data',0,1);
% train_data = train_data';
% test_data = mapminmax(test_data',0,1);
% test_data = test_data';
% validation = mapminmax(validation',0,1);
% validation = validation';
%[train_data,test_data] = scaleForSVM(train_data,test_data);


train_bar_dsp = train_data * bar_dsp_coef;
test_bar_dsp = test_data * bar_dsp_coef;
validation_bar_dsp = validation * bar_dsp_coef;

[train_bar_dsp,validation_bar_dsp,  test_bar_dsp,bar_dsp] = SVM_2_class_sp(train_bar_dsp,train_labels,validation_bar_dsp, test_bar_dsp,test_labels, 1);


train_cntry_dsp = train_data * cntry_dsp_coef;
test_cntry_dsp = test_data * cntry_dsp_coef;
validation_cntry_dsp = validation * cntry_dsp_coef;

[train_cntry_dsp,validation_cntry_dsp, test_cntry_dsp,cntry_dsp] = SVM_2_class_sp(train_cntry_dsp,train_labels,validation_cntry_dsp,test_cntry_dsp,test_labels, 4);

train_elec_dsp = train_data * elec_dsp_coef;
test_elec_dsp = test_data * elec_dsp_coef;
validation_elec_dsp = validation * elec_dsp_coef;

[train_elec_dsp,validation_elec_dsp,  test_elec_dsp,elec_dsp] = SVM_2_class_sp(train_elec_dsp,train_labels,validation_elec_dsp, test_elec_dsp,test_labels, 5);

load('Pandora_vdsp');
% train_data = mapminmax(train_data',0,1);
% train_data = train_data';
% test_data = mapminmax(test_data',0,1);
% test_data = test_data';
% validation = mapminmax(validation',0,1);
% validation = validation';
%[train_data,test_data] = scaleForSVM(train_data,test_data);


train_blues_vdsp = train_data * blues_vdsp_coef;
test_blues_vdsp = test_data * blues_vdsp_coef;
validation_blues_vdsp = validation * blues_vdsp_coef;

[train_blues_vdsp, validation_blues_vdsp, test_blues_vdsp,blues_vdsp] = SVM_2_class_sp(train_blues_vdsp,train_labels,validation_blues_vdsp ,test_blues_vdsp,test_labels,2);

train_class_vdsp = train_data * class_vdsp_coef;
test_class_vdsp = test_data * class_vdsp_coef;
validation_class_vdsp = validation * class_vdsp_coef;

[train_class_vdsp,validation_class_vdsp,  test_class_vdsp,class_vdsp] = SVM_2_class_sp(train_class_vdsp,train_labels,validation_class_vdsp, test_class_vdsp,test_labels,3);

blocksize = 128;
hopsize = 32;

train_data = readfile(trainfile);
test_data = readfile(testfile);

validation_data = readfile(validationfile);
% train_data = [train_data;validation_data];
% train_labels = [train_labels;validation_labels];

train2 = [];
train2_labels = [];

% 
% train_bar_dsp = mapminmax(train_bar_dsp',0,1);
% train_bar_dsp = train_bar_dsp';
% test_bar_dsp = mapminmax(test_bar_dsp',0,1);
% test_bar_dsp = test_bar_dsp';
% validation_bar_dsp = mapminmax(validation_bar_dsp',0,1);
% validation_bar_dsp = validation_bar_dsp';
% 
% train_blues_vdsp = mapminmax(train_blues_vdsp',0,1);
% train_blues_vdsp = train_blues_vdsp';
% test_blues_vdsp = mapminmax(test_blues_vdsp',0,1);
% test_blues_vdsp = test_blues_vdsp';
% validation_blues_vdsp = mapminmax(validation_blues_vdsp',0,1);
% validation_blues_vdsp = validation_blues_vdsp';
% 
% train_class_vdsp = mapminmax(train_class_vdsp',0,1);
% train_class_vdsp = train_class_vdsp';
% test_class_vdsp = mapminmax(test_class_vdsp',0,1);
% test_class_vdsp = test_class_vdsp';
% validation_class_vdsp = mapminmax(validation_class_vdsp',0,1);
% validation_class_vdsp = validation_class_vdsp';
% 
% train_cntry_dsp = mapminmax(train_cntry_dsp',0,1);
% train_cntry_dsp = train_cntry_dsp';
% test_cntry_dsp = mapminmax(test_cntry_dsp',0,1);
% test_cntry_dsp = test_cntry_dsp';
% validation_cntry_dsp = mapminmax(validation_cntry_dsp',0,1);
% validation_cntry_dsp = validation_cntry_dsp';
% 
% train_elec_dsp = mapminmax(train_elec_dsp',0,1);
% train_elec_dsp = train_elec_dsp';
% test_elec_dsp = mapminmax(test_elec_dsp',0,1);
% test_elec_dsp = test_elec_dsp';
% validation_elec_dsp = mapminmax(validation_elec_dsp',0,1);
% validation_elec_dsp = validation_elec_dsp';
% 
% train_jazz_sp = mapminmax(train_jazz_sp',0,1);
% train_jazz_sp = train_jazz_sp';
% test_jazz_sp = mapminmax(test_jazz_sp',0,1);
% test_jazz_sp = test_jazz_sp';
% validation_jazz_sp = mapminmax(validation_jazz_sp',0,1);
% validation_jazz_sp = validation_jazz_sp';
% 
% train_metal_sp = mapminmax(train_metal_sp',0,1);
% train_metal_sp = train_metal_sp';
% test_metal_sp = mapminmax(test_metal_sp',0,1);
% test_metal_sp = test_metal_sp';
% validation_metal_sp = mapminmax(validation_metal_sp',0,1);
% validation_metal_sp = validation_metal_sp';
% 
% train_rap_sp = mapminmax(train_rap_sp',0,1);
% train_rap_sp = train_rap_sp';
% test_rap_sp = mapminmax(test_rap_sp',0,1);
% test_rap_sp = test_rap_sp';
% validation_rap_sp = mapminmax(validation_rap_sp',0,1);
% validation_rap_sp = validation_rap_sp';
% 
% train_rock_scp = mapminmax(train_rock_scp',0,1);
% train_rock_scp = train_rock_scp';
% test_rock_scp = mapminmax(test_rock_scp',0,1);
% test_rock_scp = test_rock_scp';
% validation_rock_scp = mapminmax(validation_rock_scp',0,1);
% validation_rock_scp = validation_rock_scp';
% 
% train_rom_sp = mapminmax(train_rom_sp',0,1);
% train_rom_sp = train_rom_sp';
% test_rom_sp = mapminmax(test_rom_sp',0,1);
% test_rom_sp = test_rom_sp';
% validation_rom_sp = mapminmax(validation_rom_sp',0,1);
% validation_rom_sp = validation_rom_sp';

% train_bar_dsp = [train_bar_dsp;validation_bar_dsp];
% train_blues_vdsp = [train_blues_vdsp;validation_blues_vdsp];
% train_class_vdsp = [train_class_vdsp;validation_class_vdsp];
% train_cntry_dsp = [train_cntry_dsp;validation_cntry_dsp];
% train_elec_dsp = [train_elec_dsp;validation_elec_dsp];
% train_jazz_sp = [train_jazz_sp;validation_jazz_sp];
% train_metal_sp = [train_metal_sp;validation_metal_sp];
% train_rap_sp = [train_rap_sp; validation_rap_sp];
% train_rock_scp = [train_rock_scp; validation_rock_scp];
% train_rom_sp = [train_rom_sp; validation_rom_sp];


for i = 1:size(train_data,1)
    [pp,p] = max(train_data(i,:));
    if p == 1
        [temp,~,prob] = svmpredict(train_labels(i),train_bar_dsp(i,:),bar_dsp,'-b 1');
    elseif p == 2
        [temp,~,prob] = svmpredict(train_labels(i),train_blues_vdsp(i,:),blues_vdsp,'-b 1');
    elseif p == 3
        [temp,~,prob] = svmpredict(train_labels(i),train_class_vdsp(i,:),class_vdsp,'-b 1');
    elseif p == 4
        [temp,~,prob] = svmpredict(train_labels(i),train_cntry_dsp(i,:),cntry_dsp,'-b 1');
    elseif p == 5
        [temp,~,prob] = svmpredict(train_labels(i),train_elec_dsp(i,:),elec_dsp,'-b 1');
    elseif p == 7
        [temp,~,prob] = svmpredict(train_labels(i), train_jazz_sp(i,:),jazz_sp,'-b 1');
    elseif p == 8
        [temp,~,prob] = svmpredict(train_labels(i), train_metal_sp(i,:),metal_sp,'-b 1');
    elseif p == 6
        [temp,~,prob] = svmpredict(train_labels(i), train_hip_sp(i,:),hip_sp,'-b 1');
    elseif p == 9
        [temp,~,prob] = svmpredict(train_labels(i), train_rock_scp(i,:),rock_scp,'-b 1');
    else 
        [temp,~,prob] = svmpredict(train_labels(i), train_rom_sp(i,:),rom_sp,'-b 1');
    end
    if temp == 0 || (temp ~= 0 && exp(pp) < mean(exp(train_data(i,:))) * 2 && prob(1)<0.8)
        train2 = [train2,i];
        train2_labels = [train2_labels;train_labels(i)];
    end
end

load('mfcc_info');

train_mfcc = train_mfcc_x;
test_mfcc = test_mfcc_x;
validation_mfcc = validation_mfcc_x;


%train_mfcc = [train_mfcc;validation_mfcc];

load('HPSS_mfcc_info');
train_HPSS = train_mfcc_x;
test_HPSS = test_mfcc_x;
validation_HPSS = validation_mfcc_x;

load('lfp_hopsize_pca');

clear train_mfcc_x;
clear test_mfcc_x;
clear validation_mfcc_x;

%train_HPSS = [train_HPSS;validation_HPSS];
train_x = [];
train_y = [];
len1 = length(train_mfcc);
len2 = length(validation_mfcc);

for i = 1: len1 + len2
    if isempty(find(train2 == i, 1 )) && i<=len1
        continue;
    end
    if i <= len1
        tmp = train_sample{i};
      len = size(train_mfcc{i},1);
      train_mf = [];
      for j = 1:hopsize:len
          if j + blocksize <= len
              temp = [train_mfcc{i}(j:j+blocksize,:),train_HPSS{i}(j:j+blocksize,1:20)];
              m = mean(temp);
              var = std(temp);
              dif = diff(temp);
              dif2 = diff(dif);
              m_dif2 = mean(dif2);
              var_dif2 = std(dif2);
              m_diff = mean(dif);
              m_var = std(dif);
              com = [m,var,m_var,var_dif2];
              train_mf = [train_mf;com];
              train_y = [train_y;train_labels(i)];
          end
        end
        train_mf = [train_mf,tmp(1:size(train_mf,1),1:40)];
    else 
      k = i - len1;
      len = size(validation_mfcc{k},1);
      train_mf = [];
        tmp = validation_sample{k};
      for j = 1:hopsize:len
          if j + blocksize <= len
              temp = [validation_mfcc{k}(j:j+blocksize,:),validation_HPSS{k}(j:j+blocksize,1:20)];
              m = mean(temp);
              var = std(temp);
              dif = diff(temp);
              dif2 = diff(dif);
              m_dif2 = mean(dif2);
              var_dif2 = std(dif2);
              m_diff = mean(dif);
              m_var = std(dif);
              com = [m,var,m_var,var_dif2];
              train_mf = [train_mf;com];
              train_y = [train_y;validation_labels(k)];
            end
        end
        
        train_mf = [train_mf,tmp(1:size(train_mf,1),1:40)];
    end

    train_x = [train_x;train_mf];
        
end
clear train_mfcc;
clear train_HPSS;

train_mean = mean(train_x);
train_std = std(train_x);
for i = 1:size(train_x,2)
   train_x(:,i) = (train_x(:,i)-train_mean(i))/train_std(i);
end
%


model = svmtrain(train_y,train_x);

correct = 0 ;



for i = 1:length(test_labels)
    [pp,p] = max(test_data(i,:));
    if p == 1
        [temp,~,prob] = svmpredict(p,test_bar_dsp(i,:),bar_dsp,'-b 1');
    elseif p == 2
        [temp,~,prob] = svmpredict(p,test_blues_vdsp(i,:),blues_vdsp,'-b 1');
    elseif p == 3
        [temp,~,prob] = svmpredict(p,test_class_vdsp(i,:),class_vdsp,'-b 1');
    elseif p == 4
        [temp,~,prob] = svmpredict(p,test_cntry_dsp(i,:),cntry_dsp,'-b 1');
    elseif p == 5
        [temp,~,prob] = svmpredict(p,test_elec_dsp(i,:),elec_dsp,'-b 1');
    elseif p == 7
        [temp,~,prob] = svmpredict(p, test_jazz_sp(i,:),jazz_sp,'-b 1');
    elseif p == 8
        [temp,~,prob] = svmpredict(p, test_metal_sp(i,:),metal_sp,'-b 1');
    elseif p ==6
        [temp,~,prob] = svmpredict(p, test_hip_sp(i,:),hip_sp,'-b 1');
    elseif p == 9
        [temp,~,prob] = svmpredict(p, test_rock_scp(i,:),rock_scp,'-b 1');
    else 
        [temp,~,prob] = svmpredict(p, test_rom_sp(i,:),rom_sp,'-b 1');
    end
    if temp == 0% || (temp ~=0 && exp(pp) < mean(exp(test_data(i,:))) * 2 && prob(1) < 0.8)
      len = size(test_mfcc{i},1);
      test_input = [];
      test_label = [];
        tmp = test_sample{i};
      for j = 1:hopsize:len
          if j + blocksize <= len
              temp = [test_mfcc{i}(j:j+blocksize,:),test_HPSS{i}(j:j+blocksize,1:20)];
              m = mean(temp);
              var = std(temp);
              dif = diff(temp);
              dif2 = diff(dif);
              m_dif2 = mean(dif2);
              var_dif2 = std(dif2);

              m_diff = mean(dif);
              m_var = std(dif);
              com = [m,var,m_var,var_dif2];
              test_input = [test_input;com];
              test_label = [test_label;test_labels(i)];
          end
        end
        test_input = [test_input,tmp(1:size(test_input,1),1:40)];
        test_input = z_score(test_input,train_mean,train_std);
        predict = svmpredict(test_label,test_input,model);
        p = mode(predict) ; 
    end
    if p == test_labels(i)
        correct = correct + 1 ;
    end
    cf(p,test_labels(i)) = cf(p,test_labels(i)) + 1;

end


cr = correct / length(test_labels);

end
