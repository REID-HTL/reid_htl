function [M,beta,newf,oldf] = FindmetricHTL(Xa,Xb,idxa,idxb,models,lmbda)

SZ_a= size(Xa);
N_a= SZ_a(1);
SZ_b= size(Xb);
N_b= SZ_b(1);
S_f=SZ_a(2);
unique_idx = unique(idxa);

%Diff_ab= zeros(N_a*N_b,S_f);
X_sim = [];
X_dsim = [];

disp('Calculating feature differences..');

vectors = {idxb,idxa};
indices =  cellfun(@(v) 1:numel(v), vectors, 'UniformOutput', false);
combindices = allcomb(indices{:});
IDa = idxa(combindices(:, 2));
IDb = idxb(combindices(:, 1));
Eq_pos = IDa == IDb;
pos_pos = combindices(Eq_pos,:);
X_sim = Xb(pos_pos(:,1),:)-Xa(pos_pos(:,2),:);
nPos = length(X_sim);

diff_pos = setdiff(1:length(combindices), Eq_pos);
neg_pos = combindices(diff_pos, :);

X_dsim = Xb(neg_pos(:,1),:)-Xa(neg_pos(:,2),:);
X_dsim = X_dsim(randperm(size(X_dsim, 1)), :);

if nPos > length(X_dsim)
    X_dsim = X_dsim(1:nPos,:);
end

[M,beta,newf,oldf]= HTLgrad_desc( X_sim, X_dsim,models,lmbda);

