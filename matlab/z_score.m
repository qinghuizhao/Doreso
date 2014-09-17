function input = z_score(input,i_mean,i_std)

for i = 1:size(input,2)
	input(:,i) = (input(:,i) - i_mean(i))/i_std(i);
end
end

