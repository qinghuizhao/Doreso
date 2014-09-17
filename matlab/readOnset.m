function onset = readOnset(input)

filelists = getAllFiles(input);
pitch  = [] ;
for i = 1:length(filelists)
    file = fopen(filelists{i},'r');
    count = 1;
    while ~feof(file)
        line = fgetl(file);
        count = count + 1;
        if count ~= 3
            
            continue;
        else
            line = line(14:end-1); 
            line = str2num(line);
        end
    end
  
    onset{i} = line;
end

end
