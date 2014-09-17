function output = readfile(filename)
file = fopen(filename,'r');
output = [];
while ~feof(file)
	line = fgetl(file);
	line = str2num(line);
	output = [output;line];
end
%{
count = 0 ; 
temp = [];
while ~feof(file)
    if count ~= num
        count = count + 1;
        
        fileline = fgetl(file);
        fileline = str2num(fileline);
        temp = [temp; fileline];
    else
        count = 0 ;
        m = mean(temp);
        s= std(temp);
        combine = [m,s];
        output = [output;combine];
        temp = [];
    end
end
%}
fclose(file);
end

