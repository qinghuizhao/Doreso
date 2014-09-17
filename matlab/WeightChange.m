function weight = WeightChange(predict,label,weight)

for i = 1:length(predict)
    if predict(i) ~= label(i)
        weight(i) = weight(i) + 1;
    end
end
end

