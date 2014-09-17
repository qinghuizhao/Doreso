function [sample,label] = resample(input,weight,num,labels)

len = sum(weight);
label = zeros(num,1);
sample = zeros(num,size(input,2));
for i = 1:num
    seed = randi(len);
    for j = 1:length(weight)
        if seed <= sum(weight(1:j))
            break;
        end
    end
    
    label(i) = labels(j);
    sample(i,:) = input(j,:);
end

end
 
