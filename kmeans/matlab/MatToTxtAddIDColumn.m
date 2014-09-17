clear
clc
tic
filename = 'STIP_samp_100w.mat';
   matrix = importdata(filename);
   [m,n] = size(matrix)
   matrix2 = zeros(m,1);
   for i = 1:m
       matrix2(i,1) = i;
   end
   matrix2 = [matrix2,matrix];
   [~, name, ~] = fileparts(filename);
   eval(['save ',name,'.txt matrix2 -ascii;']);
%  save data.txt matrix2 -ascii
 size(matrix2)
toc
clear