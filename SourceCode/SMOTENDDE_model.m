function perf = SMOTENDDE_model(train, test, learner, testLOC)
%SMOTENDDE_MODEL Summary of this function goes here: Implement the SMOTENDDE model.
%   Detailed explanation goes here

rand('seed',0);

balancedTrain = train;
if sum(train(:,end)>0) < sum(train(:,end)==0)
    balancedTrain = SMOTENDDE(train, learner); % Call for self-defined function - SMOTENDDE
end
balTrainX = balancedTrain(:,1:end-1);
balTrainY = balancedTrain(:,end);
switch learner
    case 'LR'
        mdl = fitlm(balTrainX, balTrainY);
        predY = predict(mdl, test(:,1:end-1)); %
    case 'CART'
        mdl = fitrtree(balTrainX,balTrainY);
        predY = predict(mdl, test(:,1:end-1)); %
    case 'RF'
        nTrees = 20;
        mdl = TreeBagger(nTrees, balTrainX, balTrainY, 'Method','regression');
        predY = predict(mdl, test(:,1:end-1)); %
    case 'BRR'
        mdl = py.sklearn.linear_model.BayesianRidge();
        mdl.fit(balTrainX,balTrainY);
        predY = mdl.predict(test(:,1:end-1));
        predY = (predY.data.double)';
    case 'GBR'
        mdl = py.sklearn.ensemble.GradientBoostingRegressor('ls', 0.1, int64(20), 1.0, 'friedman_mse', int64(2), int64(1), 0.0, int64(3), 0.0, py.None, py.None, int64(0));
        mdl.fit(balTrainX,balTrainY);
        predY = mdl.predict(test(:,1:end-1));
        predY = (predY.data.double)';
    case 'SVR'
        rand('seed',1);
        mdl = fitrsvm(balTrainX,balTrainY,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);
        predY = predict(mdl, test(:,1:end-1));
    case {'pr','nbin','zip'}
        mdl = py.CountModelFun.count_model(py.numpy.array([balTrainX,balTrainY]), py.numpy.array([test]), learner);
        predY = mdl{1}.double;
end

% Ensure Y is non-negative integer
predY = round(predY);
predY(predY<0) = 0;

% Prediction
perf = RegPerformance(test(:,end), predY, testLOC);
end

