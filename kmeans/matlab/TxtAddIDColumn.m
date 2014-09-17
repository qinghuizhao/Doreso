clear
clc
tic
filename = 'stip_150_120.txt';
   matrix = importdata(filename);
   [m,n] = size(matrix)
   matrix2 = zeros(m,1);
   for i = 1:m
       matrix2(i,1) = i;
   end
   matrix2 = [matrix2,matrix];
   eval(['save ',filename,'.txt matrix2 -ascii;']);
%  save data.txt matrix2 -ascii
 size(matrix2)
toc