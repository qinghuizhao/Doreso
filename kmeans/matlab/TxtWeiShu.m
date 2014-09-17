clear
clc
tic
filename = 'stip_150_120.txt';
%     matrix = importdata(filename);
    matrix = load(filename);
disp (filename)
 [m,n] = size(matrix)
toc
clear