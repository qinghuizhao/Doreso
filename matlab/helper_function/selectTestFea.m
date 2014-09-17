function testset = selectTestFea(testset,new_feature)

for i = 1:length(testset)
    testset{i} = testset{i}(:,new_feature);
end

end

