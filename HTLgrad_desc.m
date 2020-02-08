function [M,beta,newf,oldf]= HTLgrad_desc( X_sim, X_dsim,models,lmbda)

SZ=size(X_sim);
S_f= SZ(2);

l= length(models);
beta = ones(l,1);
M = eye(S_f);
%M = rand(S_f);
%M = M*M';

% Set hyperparameters
z=zeros(l,1);
alpha=0.2;
gamma=0.1;
theta=0;
ths2=40; 

C_S= X_sim'*X_sim / size(X_sim,1);
C_D= X_dsim'*X_dsim / size(X_dsim,1);
 
func = @(M,beta) trace(M*C_S) + lmbda * norm((compute_reg(M,models,beta)),'fro')^2 + theta*norm(beta)^2;
gradfunc= @(M,beta) C_S +2*lmbda*(compute_reg(M,models,beta));

oldf= func(M,beta);
epsilon=10^-1;
newf= 10^15;

% Run optimization
while(abs((newf-oldf)/oldf)>=0.0001)
  
    oldf= newf;
    M=M-alpha*gradfunc(M,beta);
    temp=M;
    oldM=M;
    newM= randn(size(M));

    while(max(max(abs(oldM-newM)))>=0.01)  

         oldM=temp;
         x= (ths2-trace(oldM*C_D))/norm(C_D,'fro')^2;
         psi= max(0,x);
         oldM =oldM+psi*C_D;

         [V,S]=eig(oldM);
         for i= 1: length(oldM)
             S(i,i)= max(0,S(i,i));
         end
         S=real(S);
         V=real(V);
         newM= (V*S*V');
         temp=newM;
    end
    M=oldM;
   
    beta= beta -gamma*compute_grad_beta(M,models,beta,theta)/norm(compute_grad_beta(M,models,beta,theta));
    beta=max((beta/(norm(beta))),z);

    newf= func(M,beta);
   
    fprintf('similar and dissimilar pair distances and diffobjective are %d ,%d ,%d \n',trace(M*C_S),trace(M*C_D),abs((newf-oldf)/1));
end

end



         
    
