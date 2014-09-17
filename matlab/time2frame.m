function frame = time2frame(time,fs,hopsize)

sample = round(time*fs);
frame = ceil(sample / hopsize);

end