clear
clc
tic
 filename = 'SIFT_samp_100w.txt.cluster_centres';
 matrix = importdata(filename);
 matrix2 = matrix(:,2:end);
 eval(['save ',filename,'.mat matrix2;']);
% save centers.mat matrix2
[m,n] = size(matrix2)
toc
clear