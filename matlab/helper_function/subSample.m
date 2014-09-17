function sample = subSample(input,num)

len = length(input);

idx = randperm(len);
len = round(num*len);

for i = 1:len
    sample{i} = input{idx(i)};
end

end
