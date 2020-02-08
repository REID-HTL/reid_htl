%% Script to perform HTL on RAiD dataset - average accuracy of 10 iterations is reported
clc; clear all;

iterations = 1;     % Number of train/test iterations to run
pca_dim = 100;      % Number of feature dimensions to use after PCA
lmbda = 0.001;      % Hyper parameter
rng(0);
dataset = 'raid';   % Should be 'ward' or 'raid'

% Load LOMO features and camera info
feat_path  = strcat('./features/',dataset,'_lomo.mat');
load(feat_path,'features');
load(strcat(dataset,'_splits.mat'));

info = load(strcat(dataset,'_info.mat'));
info = info.info;
unique_id = unique(info.personid);

if strcmp(dataset,'raid')
    unique_id = unique_id(unique_id ~= 8 & unique_id ~= 34);    % Remove persons not present in all cameras
    num_cam = 4;
else
    num_cam = 3;
end

% Results for all 10 iterations
acc_htl_total = {};
acc_k_total = {};
acc_best_s_total = {};
acc_all_s_total = {};

for i=1:iterations

    % Run HTL
    [acc,acc_k,acc_all_s,acc_best_s,cmc_avg_htl,cmc_avg_k,cmc_avg_best_s,cmc_avg_all_s] = train_HTL(features,info,source_id(i,:),target_id(i,:),test_ids(i,:),lmbda,pca_dim);
   
    % Camwise acc for htl
    for j=1:num_cam-1
        cmc_1(j,:) = acc{j};
        cmc_2(j,:) = acc{j+1*(num_cam-1)};
        cmc_3(j,:) = acc{j+2*(num_cam-1)};
        if strcmp(dataset,'raid')
            cmc_4(j,:) = acc{j+3*(num_cam-1)};
        end
    end
    
    cmc_htl_all(i,:) = cmc_avg_htl;   % HTL
    cmc_k_all(i,:) = cmc_avg_k;       % KISSME
    cmc_best_s_all(i,:) = cmc_avg_best_s; % Best source
    cmc_all_s_all(i,:) = cmc_avg_all_s;   % Average source
    
    acc_htl_all{end+1} = acc;
    acc_k_all{end+1} = acc_k;
    acc_best_s_all{end+1} = acc_best_s;
    acc_all_s_all{end+1} = acc_all_s;
end
cmc_htl = mean(cmc_htl_all);
cmc_k = mean(cmc_k_all);
cmc_best_s = mean(cmc_best_s_all);
cmc_all_s = mean(cmc_all_s_all);