function cmcs = calcCMC(idxtest, ranks)

for i=1: numel(ranks)
    result = zeros(1, numel(idxtest));
    for j=1:numel(idxtest)
        result(ranks{i}(j,:)==idxtest(j)) = result(ranks{i}(j,:)==idxtest(j)) + 1;
    end
    
    tmp = 0;
    for counter=1:length(result)
        result(counter) = result(counter) + tmp;
        tmp = result(counter);
    end
    result = result/ numel(idxtest);
    cmcs{i} = result;
end