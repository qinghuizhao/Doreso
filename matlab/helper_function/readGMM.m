addpath('../data/');

train_data = readfile('train.txt');

test_data = readfile('test.txt');


save ../data/GMM_prime    train_data test_data
