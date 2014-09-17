clear
clc
tic
filename = 'STIP_ind_100W.txt.cluster_centres.mat';
   matrix = importdata(filename);
eval(['save ',filename,'.txt matrix -ascii;']);
% save data.txt matrix -ascii
[m,n] = size(matrix)
toc
clear