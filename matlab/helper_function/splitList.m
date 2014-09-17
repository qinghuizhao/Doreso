function splitList(list,trainlist,validationlist)

f = fopen(list,'r');
count = 1;
while ~feof(f)
    line = fgetl(f);
    buffer{count} = line;
    count = count + 1;
end
len = length(buffer);
idx = randperm(len);

train_buffer = buffer(idx(1:round(len/2)));
validation_buffer = buffer(idx(round(len/2):end));

fclose(f);

f = fopen(trainlist,'w');

for i = 1:length(train_buffer)
    fprintf(f,'%s\n',train_buffer{i});
end

fclose(f);

f= fopen(validationlist,'w');
for i = 1:length(validation_buffer)
    fprintf(f,'%s\n',validation_buffer{i});
end

fclose(f);

end