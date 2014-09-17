%
%count = 1;
%input = [];
%for i = 1:length(lfp2)
%    temp = lfp2{i};
%    for j = 1:length(temp)
%        temp{j} = temp{j} * cof;
%        input = [input;temp{j}];
%       % test_sample{count} = temp{j};
%        
%        count = count + 1;
%    end
%end
%
%[cof,score,la] = princomp(input);

function [train,test,validation] = doPCA(trainset,testset,validationset,num)

[cof,score,la] = princomp(trainset);

if nargin<3
    num = length(find(la>1.5));
end

cof = cof(:,1:num);

train = trainset * cof;

test = testset * cof;

validation = validationset * cof;
end

