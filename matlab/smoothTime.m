function input = smoothTime(input)

for i = 1:length(input);
    onset = input{i};
    flag = 1;
    start = 1;
    over = start + 1;
    while over<=length(onset)
        while onset(over)-onset(start) < 1 && over<length(onset)
            onset(over) = [];
        end
        start = start + 1;
        over = start + 1;
    end
    input{i} = onset;
end
end
        