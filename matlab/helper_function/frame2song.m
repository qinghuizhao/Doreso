function [train] = frame2song(trainset,blocksize,hopsize)

train = [];

for i = 1:length(trainset)
    frames = trainset{i};
    len = size(frames,1);
    temp = [];
    for j = 1:hopsize:len
        if j + blocksize <= len
            m = mean(frames(j:j+blocksize,1:13));
            var = std(frames(j:j+blocksize,1:13));
            dif = diff(frames(j:j+blocksize,1:13));
            dif2 = diff(dif);
            m_dif2 = mean(dif2);
            var_dif2 = std(dif2);
    
            m_diff = mean(dif);
            m_var = std(dif);
            com = [m,var,m_var,var_dif2];
           %com = [m,var];
            temp = [temp,com]; 
        end 
    end
    train = [train;temp];
end

end
