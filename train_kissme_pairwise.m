function [models, s_acc] = train_kissme_pairwise(source_feat,source_id,source_camid,num_cam,i)

% Split data depending on camera views
camID = linspace(1,num_cam,num_cam);
features_split = {};
trainID_split = {};
trainCAM_split = {};

for j=1:length(camID)
    if ~any(i == j)
        cam_temp = source_camid == camID(j);
        trainCAM_split{end+1} = source_camid(cam_temp);
        trainID_split{end+1} = source_id(cam_temp);
        features_split{end+1} = source_feat(:,cam_temp);
    end
end

camIDTrain = linspace(1,num_cam-numel(i),num_cam-numel(i));

% Calculate pairwise kissme metrics
params.numCoeffs = size(source_feat,1);
models = {};
dim = size(source_feat,1);
disp('Training source pairwise models..');
for m=1:length(camIDTrain)
    if m ~= length(camIDTrain)
        for j=(m+1):length(camIDTrain)
            %disp(strcat('Calculating for camera ',string(m), ' and ',string(j)));
            train_idx = [trainID_split{m};trainID_split{j}];
            train_cam = [trainCAM_split{m};trainCAM_split{j}];
            
            train_features = [features_split{m} features_split{j}];
            [idxa,idxb,flag] = gen_train_sample_kissme(train_idx, train_cam);
            s = learnPairwise(LearnAlgoKISSME(params),train_features,idxa,idxb,logical(flag));
           
            models{end+1} = s.M;
        end
    end
end

end