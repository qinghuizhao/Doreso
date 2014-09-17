addpath('./data/');

load('Pandora_scp');

[train_scp,test_scp,validation_scp] = doPCA(train_data,test_data,validation,24);

load('Pandora_dsp');

[train_dsp,test_dsp,validation_dsp] = doPCA(train_data,test_data,validation,40);

load('Pandora_vdsp');

[train_vdsp,test_vdsp,validation_vdsp] = doPCA(train_data,test_data,validation,100);


load('Pandora_sp');


[train_sp,test_sp,validation_sp] = doPCA(train_data,test_data,validation,24);

save ./data/scp_pca train_scp test_scp validation_scp;
save ./data/dsp_pca train_dsp test_dsp validation_dsp;
save ./data/vdsp_pca train_vdsp test_vdsp validation_vdsp;
save ./data/sp_pca train_sp test_sp validation_sp;

% load('gyt_cp_lfp');
% 
% [train_cp,test_cp] = doPCA(cp1,cp2,40);
% 
% [train_lfp,test_lfp] = doPCA(lfp1,lfp2,40);
% 
% save ./data/cp_pca train_cp test_cp train_data_labels test_data_labels;
% 
% save ./data/lfp_pca train_lfp test_lfp;

