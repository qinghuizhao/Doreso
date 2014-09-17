function SVM_GMM_final(trainlist,testlist,output)

train_x = [];
train_y = [];

mfcc_dir = './data/radar/feats/mfcc/';
hpss_dir = './data/radar/feats/hm/';
lfp_dir = './data/radar/feats/lfp/';
blocksize = 128;
hopsize = 32;

f = fopen(trainlist,'r');

gmm_block = [];

while ~feof(f)
    line = fgetl(f);
    line = deblank(line);
    line = regexp(line,'\t','split');
    lines = line(1);
    label = line(2);
    label = label{1};
    label = label2num(label);
    lines = lines{1};
    [path,filename,ext] = fileparts(lines);
    mfcc_name = [mfcc_dir,filename,ext,'.mat'];
    mfcc = load(mfcc_name);   
    len = size(mfcc.mfcc,1);
    for i = 1:hopsize:len
        
       if i + blocksize <= len
            temp = [mfcc.mfcc(i:i+blocksize,2:end)];
            m = mean(temp);
            var = std(temp);
            dif = diff(temp);
            dif_m = mean(dif);
            com = [m,var,dif_m];
            gmm_block = [gmm_block;com];
            train_y = [train_y;label];
       end
    end
end
gmm_block = double(gmm_block);
gmm_m = mean(gmm_block);
gmm_var = std(gmm_block);
gmm_block = z_score(gmm_block,gmm_m,gmm_var);

for j = 1:10
    train_class{j} = [];
end

for i = 1:length(train_y)
    train_class{train_y(i)} = [train_class{train_y(i)};gmm_block(i,:)];
end



for i = 1:10
    obj{i} = gmdistribution.fit(train_class{i},5,'CovType','diagonal');
end 
    
gmm_post = [];
for i = 1:10
    post = posterior(obj{i},gmm_block);
    gmm_post = [gmm_post,post];
end

fseek(f,0,-1);
while ~feof(f)
    line = fgetl(f);
    line = deblank(line);
    line = regexp(line,'\t','split');
    lines = line(1);
    label = line(2);
    label = label{1};
    label = label2num(label);
    lines = lines{1};
    [path,filename,ext] = fileparts(lines);
    mfcc_name = [mfcc_dir,filename,ext,'.mat'];
    hpss_name = [hpss_dir,filename,ext,'.mat'];
    lfp_name = [lfp_dir,filename,ext,'.mat'];

    mfcc = load(mfcc_name);   
    hpss = load(hpss_name);
    lfp = load(lfp_name);

    train_tmp = [];
    for i = 1:hopsize:size(mfcc.mfcc,1)
        if i + blocksize <= size(mfcc.mfcc,1)
            temp = [mfcc.mfcc(i:i+blocksize,2:end),hpss.mfcc(i:i+blocksize,2:end)];
            dif = diff(temp);
            dif2 = diff(dif);
            m = mean(temp);
            var = std(temp);
            var_dif = std(dif);
            var_dif2 = std(dif2);
            com = [m,var,var_dif,var_dif2];
            train_tmp = [train_tmp;com];
            %train_y = [train_y;label];
        end
    end
    train_x = [train_x;train_tmp,lfp.lfp_single(1:size(train_tmp,1),1:40)];
end
fclose(f);

train_x = [train_x,gmm_post];

train_x = double(train_x);
tr_m = mean(train_x);
tr_var = std(train_x);

train_x = z_score(train_x,tr_m,tr_var);

model = svmtrain(train_y,train_x);
outputfile = fopen(output,'w');
f = fopen(testlist,'r');
correct = 0 ;
while ~feof(f)
    line = fgetl(f);
    line = deblank(line);
    line = regexp(line,'\t','split');
    lines = line(1);
    lines = lines{1};
    [path,filename,ext] = fileparts(lines);
    %[path,filename,ext] = fileparts(line);

    mfcc_name = [mfcc_dir,filename,ext,'.mat'];
    hpss_name = [hpss_dir,filename,ext,'.mat'];
    lfp_name = [lfp_dir,filename,ext,'.mat'];

    mfcc = load(mfcc_name);
    hpss = load(hpss_name);
    lfp = load(lfp_name);

    test_input = [];
    test_gmm = [];
    len = size(mfcc.mfcc,1);
    for i = 1:hopsize:len
        if i + blocksize  <= len
            gmm_temp = mfcc.mfcc(i:i+blocksize,2:end);
            temp = [mfcc.mfcc(i:i+blocksize,2:end),hpss.mfcc(i:i+blocksize,2:end)];
            gmm_dif = diff(gmm_temp);
            dif = diff(temp);
            dif2 = diff(dif);
            m = mean(temp);
            gmm_mean = mean(gmm_temp);
            gmm_std = std(gmm_temp);
            gmm_dif_mean = mean(gmm_dif);
            var = std(temp);
            var_dif = std(dif);
            var_dif2 = std(dif2);
            gmm_com = [gmm_mean,gmm_std,gmm_dif_mean];
            test_gmm = [test_gmm;gmm_com];
            com = [m,var,var_dif,var_dif2];
            test_input = [test_input;com];
        end
    end
    test_input = [test_input,lfp.lfp_single(1:size(test_input,1),1:40)];
    test_post = [];
    for i = 1:10
         post = posterior(obj{i},test_gmm);
         test_post = [test_post,post];
    end
    test_input = [test_input,test_post];
    test_input = double(test_input);
    test_input = z_score(test_input,tr_m,tr_var);
    [predict] = svmpredict(ones(size(test_input,1),1),test_input,model);
    prediction = num2label(mode(predict));
    fprintf(outputfile,'%s\t%s\n',lines,prediction);
end
disp(correct);
fclose(outputfile);
fclose(f);

end



