function [acc,acc_k,acc_all_s,acc_best_s,cmc_avg_htl,cmc_avg_k,cmc_avg_best_s,cmc_avg_all_s] = train_HTL(features,info,source_id,target_id,test_id,lmbda,pca_dim)

% Number of cameras in dataset
num_cam = numel(unique(info.camid));
dim = size(features,1);

% Split features into source, target and test
source = ismember(info.personid,source_id);
source_features_temp = features(:,source);
source_features_norm = source_features_temp ./ repmat(sqrt(sum(source_features_temp.*source_features_temp)), dim, 1);
[ux_source,u_,m_] = applypca2(source_features_norm);
source_features = ux_source(1:pca_dim,:);

camid_source = info.camid(source);
personid_source = info.personid(source);

target = ismember(info.personid,target_id);
target_features_temp = features(:,target);
target_features_norm = target_features_temp ./ repmat(sqrt(sum(target_features_temp.*target_features_temp)), dim, 1);
ux_target = u_'*(target_features_norm-repmat(m_,1,size(target_features_norm,2)));
target_features = ux_target(1:pca_dim,:);

camid_target = info.camid(target);
personid_target = info.personid(target);

test = ismember(info.personid,test_id);
test_features_temp = features(:,test);
test_features_norm = test_features_temp ./ repmat(sqrt(sum(test_features_temp.*test_features_temp)), dim, 1);
ux_test = u_'*(test_features_norm-repmat(m_,1,size(test_features_norm,2)));
test_features = ux_test(1:pca_dim,:);

camid_test = info.camid(test);
personid_test = info.personid(test);

% To save camwise results
acc = {};
acc_k = {};
acc_best_s = {};
acc_all_s = {};

rank_temp = 0;

%%
counter = 1;

params.numCoeffs = pca_dim;
% Run through each camera as target and use other as source
for i=1:num_cam
    disp(strcat('Camera',' ', string(i), ' is now target..'));
    
    % Train pairwise source kissme metrics using source training data
    s_models = train_kissme_pairwise(source_features,personid_source',camid_source',num_cam,i);
    disp('Pairwise models trained..')

    % Get target and test features from camera 'i'
    camid_temp = camid_target == i;
    camid_train1 = camid_target(camid_temp);
    idxa = personid_target(camid_temp);
    target_features_view1 = target_features(:,camid_temp);
    
    camid_test_temp = camid_test == i;
    idxa_test = personid_test(camid_test_temp);
    test_features_view1 = test_features(:,camid_test_temp);
    
     % Train pairwise models m_tp using target data and source models
    for j=1:num_cam
        if j ~= i
            % Get target and test features from camera 'j' (j ~= i)
            camid_temp2 = camid_target == j;
            idxb = personid_target(camid_temp2);
            camid_train2 = camid_target(camid_temp2);
            target_features_view2 = target_features(:,camid_temp2);
            
            camid_test_temp2 = camid_test == j;
            idxb_test = personid_test(camid_test_temp2);
            test_features_view2 = test_features(:,camid_test_temp2);

            train_label = [idxa';idxb'];
            train_cam = [camid_train1';camid_train2'];
            train_features_ = [target_features_view1 target_features_view2];

            [idxa_k,idxb_k,flag] = gen_train_sample_kissme(train_label, train_cam);
            s = learnPairwise(LearnAlgoKISSME(params),train_features_,idxa_k,idxb_k,logical(flag));
            
            % Train HTL for target
            [t_model,beta,newf,oldf] = FindmetricHTL(target_features_view1',target_features_view2',idxa,idxb,s_models,lmbda);
            
            % Evaluate HTL between camera i and j
            cmc_htl_temp = htl_cmc(test_features_view1,test_features_view2,idxa_test,idxb_test,t_model,1);
            cmc_k_temp = htl_cmc(test_features_view1,test_features_view2,idxa_test,idxb_test,s.M,1);
            cmc_htl(counter,:) = cmc_htl_temp;
            cmc_k(counter,:) = cmc_k_temp;
            acc{end+1} = cmc_htl_temp;
            acc_k{end+1} = cmc_k;
            
            % Evaluate average source and choose best source
            if numel(s_models) > 1
                for n=1:numel(s_models) % Using best source
                    cmc_temp = htl_cmc(test_features_view1,test_features_view2,idxa_test,idxb_test,s_models{n},1);
                    if cmc_temp(1) > rank_temp
                        cmc_best = cmc_temp;
                        rank_temp = cmc_best(1);
                    end
                    cmc_all_temp(n,:) = cmc_temp;
                end
                cmc_all_s(counter,:) = mean(cmc_all_temp);
                acc_all_s{end+1} = mean(cmc_all_temp);
            else
                cmc_temp = htl_cmc(test_features_view1,test_features_view2,idxa_test,idxb_test,s_models{1},1);
                cmc_all_s(counter,:) = cmc_temp;
                acc_all_s{end+1} = cmc_temp;
                cmc_best = cmc_temp;
            end
            
            cmc_best_s(counter,:) = cmc_best;
            acc_best_s{end+1} = cmc_best;
            
            rank_temp = 0;
            counter = counter + 1;
        end
    end
    
end

cmc_avg_htl = mean(cmc_htl);
cmc_avg_k = mean(cmc_k);
cmc_avg_best_s = mean(cmc_best_s);
cmc_avg_all_s = mean(cmc_all_s);
end
