addpath('./data/');
addpath('./helper_function');



% load('sp_pca');
% load('dsp_pca');
% load('vdsp_pca');
% load('cp_pca');
% load('scp_pca');
% load('lfp_pca');
load('mfcc_info');

[test_mfcc] = frame2song(test_mfcc_x,256,64); 

[alpha,model,norm_mean,norm_std] = SVM_boosting_new(train_mfcc_x,train_mfcc_y,1);

correct = 0;

test_mfcc = z_score(test_mfcc,norm_mean{1},norm_std{1});
%test_sp = z_score(test_sp,norm_mean{2},norm_std{2});
%test_dsp = z_score(test_dsp,norm_mean{3},norm_std{3});
%test_vdsp = z_score(test_vdsp,norm_mean{4},norm_std{4});
%test_scp = z_score(test_scp,norm_mean{5},norm_std{5});
alpha = [16,8,4,2,1];

for i = 1:length(test_mfcc_y)
    prob_final = zeros(1,10);
    prob = zeros(7,10);
    
    [p1,a1,prob(1,:)] = svmpredict(test_mfcc_y(i),test_mfcc(i,:),model{1},'-b 1');
    [p2,a1,prob(2,:)] = svmpredict(test_mfcc_y(i),test_mfcc(i,:),model{2},'-b 1');
    [p3,a1,prob(3,:)] = svmpredict(test_mfcc_y(i),test_mfcc(i,:),model{3},'-b 1');
    [p4,a1,prob(4,:)] = svmpredict(test_mfcc_y(i),test_mfcc(i,:),model{4},'-b 1');
    [p5,a1,prob(5,:)] = svmpredict(test_mfcc_y(i),test_mfcc(i,:),model{5},'-b 1');
   % [p1,a1,prob(6,:)] = svmpredict(test_mfcc_y(i),test_lfp(i,:),model{6},'-b 1');
    
   % [p1,a1,prob(7,:)] = svmpredict(test_mfcc_y(i),test_mfcc(i,:),model{7},'-b 1');
   for i = 1:length(alpha)
       prob_final = alpha(i) * prob(i,:) + prob_final;
   end
    prob_final = prob(1,:);
%    prob_final = prob(5,:);
%    prob_final = prob(2,:);
    [m,predict] = max(prob_final);
   
   if p1 == test_mfcc_y(i) || p2 == test_mfcc_y(i) || p3 == test_mfcc_y(i) || p4 == test_mfcc_y(i) || p5 == test_mfcc_y(i)  
        correct = correct + 1;
   end
end

cr = correct / length(test_mfcc_y)
