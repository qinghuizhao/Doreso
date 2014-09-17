train_cf = zeros(10,10);
val_cf = train_cf;
test_cf = train_cf;

addpath('./data/');
addpath('./helper_function/');

load('mfcc_info');

train_labels = train_mfcc_y;
test_labels = test_mfcc_y;
validation_labels = validation_mfcc_y;

train_result = readfile('train.txt');

val_result = readfile('validation.txt');

test_result = readfile('test.txt');

[~,p1] = max(train_result');
train_correct = 0 ;

for i = 1:length(p1)
    if p1(i) == train_labels(i)
       train_correct = train_correct + 1;
        
    end
    train_cf(p1(i),train_labels(i)) =  train_cf(p1(i),train_labels(i)) + 1; 
end

[~,p2] = max(val_result');

val_cor = 0 ;

for i = 1:length(p2)
    if p2(i) == validation_labels(i);
        val_cor = val_cor + 1;
    end
    
    val_cf(p2(i),validation_labels(i)) = 1+  val_cf(p2(i),validation_labels(i))  ; 
end


[~,p3] = max(test_result');

test_cor = 0 ;

for i = 1:length(p3)
    if p3(i) == test_labels(i);
        test_cor = test_cor + 1;
    end
    
    test_cf(p3(i),test_labels(i)) = 1+ test_cf(p3(i),test_labels(i)) ; 
end
train_correct = train_correct/length(p1) 
val_cor = val_cor/length(p2)
test_cor = test_cor/length(p3)
train_cf
val_cf
test_cf
    
