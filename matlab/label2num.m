function num = label2num(label)

switch label
case 'pop'
    num = 1;
case 'blues'
    num = 2;
case 'classical'
    num = 3;
case 'country'
    num = 4;
case 'disco'
    num = 5;
case 'hiphop'
    num = 6;
case 'jazz'
    num = 7;
case 'metal'
    num = 8;
case 'rock'
    num = 9;
otherwise 
    num = 10;
end

end
