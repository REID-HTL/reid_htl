function CMC2 = htl_cmc(test_feats_view1,test_feats_view2, idxa_test,idxb_test,M,mahal)

unique_test = unique(idxa_test);
num_test = numel(unique_test);
num_feats = size(test_feats_view1,1);

ux_test1 = zeros(num_feats,num_test);

ux_test2 = test_feats_view2;
for i=1:num_test
    view1_temp = test_feats_view1(:,idxa_test==unique_test(i));
    ux_test1(:,i) = mean(view1_temp,2);
end

% Calcuate Mahalnobis distance using target model
if mahal
    distTe = MahDist(M, ux_test1', ux_test2');
else
    distTe = sqdist(ux_test2, ux_test1);
    distTe = distTe';
end

k = 1;
ranks = {};
for pairCounter=1:size(distTe,1)
    distPair = distTe(pairCounter,:);  
    [tmp,idx] = sort(distPair,'ascend');
    ranks{k}(pairCounter,:) = unique(idxb_test(idx),'stable');
end

cmcs = calcCMC(unique_test, ranks);
CMC1 = cmcs{1,1};
CMC2 = CMC1*100;
%disp('Rank1   Rank5   Rank10   Rank20')
%fprintf('%2.2f,   %2.2f,   %2.2f,   %2.2f\n',CMC2(1),CMC2(5),CMC2(10),CMC2(20));

end