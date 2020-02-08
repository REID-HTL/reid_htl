function reg = compute_reg(M,models,beta)
sum= zeros(size(M));

for i=1:length(beta)
    sum=sum+beta(i)*models{i};
end
reg= M-sum;
