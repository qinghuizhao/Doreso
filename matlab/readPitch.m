function pitch = readPitch(input)

filelists = getAllFiles(input);

for i = 1:length(filelists)
    file = fopen(filelists{i},'r');
    [p,n,ext] = fileparts(filelists{i});
    if ~strcmp(ext,'.txt')
        continue;
    end
    count = 1;
    while ~feof(file)
        line = fgetl(file);
        count = count + 1;
        if count ~= 9
            continue;
        else
            line = line(17:end-1); 
            line = str2num(line);
            break;
        end
    end
    j = 1:8:length(line);
    j = [j,length(line)];
    pitch_downsample = line(j);
    pitch{i} = pitch_downsample;
end

end
