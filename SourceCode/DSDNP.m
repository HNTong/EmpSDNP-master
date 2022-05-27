function perfs = DSDNP(trainData, testData, testLOC)
%DSDNP Summary of this function goes here
%   Detailed explanation goes here
% Inputs:
%   (1) trainData - A n_tr * (d+1) matrix where the last column is defects;
%   (2) testData - A n_te * (d+1) matrix
%   (3) testLOC  - A n*1 vector denoting the Lines of Code;
% Output:
%
%
% Reference: Qiao L , Li X , Umer Q , et al. Deep learning based software
% defect prediction[J]. Neurocomputing, 2019, 385:100-110.

rand('state',0);

if~exist('testLOC','var')||isempty(testLOC)
    testLOC = 0;
end

trainX = trainData(:,1:end-1);
trainY = trainData(:,end);
numFeature = size(trainData, 2)-1;
%% Define network
layers = [
    featureInputLayer(numFeature, 'Name', 'Input Layer','Normalization','rescale-zero-one')
    fullyConnectedLayer(20, 'WeightsInitializer','glorot')
    tanhLayer
    fullyConnectedLayer(10, 'WeightsInitializer','narrow-normal')
    reluLayer
    fullyConnectedLayer(6, 'WeightsInitializer','narrow-normal')
    %batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1,'Name','Output Layer')
    regressionLayer];

%% Train network
options = trainingOptions('adam',...
    'MaxEpochs',100,...
    'MiniBatchSize',50,...
    'Shuffle','every-epoch',...
    'Verbose',0);


net = trainNetwork(trainX, trainY, layers, options);

%% Prediction
preY = predict(net, testData(:,1:end-1));
preY = round(preY);
preY(preY<0) = 0;

%% Evaluation
perfs = RegPerformance(testData(:,end), preY, testLOC);












end

