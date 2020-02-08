function gradbeta = compute_grad_beta(M,models,beta,theta)
gradbeta=zeros(length(beta),1);
ignore = 1;
for i=1:length(beta)
    gradbeta(i)= 2*theta*beta(i)+ 2*beta(i)*norm(models{i},'fro')^2-2*trace(models{i}'*(compute_reg(M,models,beta)+beta(i)*models{i}));
end
