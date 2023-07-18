function perf = SHSE(trainData, testData, learner, locIndex, useDensity, numLearners, feaRatio, insRatio)
%SHSE -- Oversampling and undersampling based sub-space ensemble algorithm
% INPUTS:
%   (1) trainData - n1_instances*(n_features+1) where the last column is the number of defects. 
%   (2) testData -  n2_instances*(n_features+1) where the last column is the number of defects.
%   (3) learner - a string, {'LR','CART','RF','BRR'}.
%   (4) locIndex - the index of LOC.
%   (5) numLearners {1,2,...} - the number of learners.
%   (6) feaRatio (0,1] - how much features are selected.
%   (7) insRatio (0,1] - how much majority class instances are selected.
% OUTPUTS:
%   perf - a strcut variable.

% Default values
if ~exist('useDensity','var')||isempty(useDensity)
    useDensity = true;
end
if ~exist('numLearners','var')||isempty(numLearners)
    numLearners = 50;
end

if ~exist('feaRatio','var')||isempty(feaRatio)
%     feaRatio = 1; % 
    feaRatio = 3/4; % 
end

if ~exist('insRatio','var')||isempty(insRatio)
    insRatio = 0.8; 
end


% Adjust insRatio
if strcmp(learner, 'LR')||strcmp(learner, 'CART')||strcmp(learner, 'RF')||strcmp(learner, 'GBR')||strcmp(learner, 'BRR')||strcmp(learner, 'SVR') || strcmp(learner, 'pr')||strcmp(learner, 'zip')||strcmp(learner, 'nbin') 
    
    ratioDef = sum(trainData(:,end)>0)/size(trainData,1);
    if ratioDef<0.5 % 
        if ratioDef+ratioDef*0.5 <= 0.5
            insRatio = (ratioDef+ratioDef*0.5)/(1-ratioDef);
        else
            insRatio = 0.5/(1-ratioDef);
        end
    else
        insRatio = 1;
    end
    
end


trainXOri = trainData(:,1:(end-1));
trainYOri = trainData(:,end);
trainLOC = trainData(:,locIndex);
trainYOriDes = trainYOri./trainLOC;

testXOri = testData(:,1:(end-1));
testYOri = testData(:,end);
testLOC = testData(:,locIndex);

K = numLearners;
selFeaNum = floor(size(trainXOri,2)*feaRatio);
modelCell = cell(1,K);
weight = zeros(2,K);
perfArra = zeros(K,6);
preES = zeros(size(testXOri,1),K);

if useDensity
    idxPos = (trainYOriDes>0);
    trainXPos = trainXOri(idxPos,:);
    trainYPos = trainYOriDes(idxPos,:);
    
    trainXNeg = trainXOri(~idxPos,:);
    trainYNeg = trainYOriDes(~idxPos,:);
else
    idxPos = (trainYOri>0);
    trainXPos = trainXOri(idxPos,:);
    trainYPos = trainYOri(idxPos,:);
    
    trainXNeg = trainXOri(~idxPos,:);
    trainYNeg = trainYOri(~idxPos,:);
end


for j0=1:K
    
    rand('seed',j0);
    idxSelFea = randperm(size(trainXOri, 2), selFeaNum);
    
%     idxSelIns = randi(size(trainXOri,1), size(trainXOri,1), 1);
%     idxSelInsPosNum = ceil(length(idxSelIns)*ratePN);
    idxSelInsPosNum = size(trainXPos,1);
    idxSelInsNegNum = floor(size(trainXNeg,1)*insRatio); % 多数类欠抽样
    
    
    idxPos = randperm(size(trainXPos,1),idxSelInsPosNum);
    idxNeg = randperm(size(trainXNeg,1),idxSelInsNegNum);
    
    newTrainX = [trainXPos(idxPos,idxSelFea);trainXNeg(idxNeg,idxSelFea)];
    newTrainY = [trainYPos(idxPos,:);trainYNeg(idxNeg,:)];
    
    % 
    matData = unique([newTrainX,newTrainY],'rows','stable');
    newTrainX = matData(:,1:(end-1));
    newTrainY = matData(:,end);
    
    % SMOTEND
    synMino = SMOTEND([newTrainX newTrainY]); % 
    if ~useDensity && ~isempty(synMino)
%         synMino = [synMino(:,1:(end-1)), round(synMino(:,end))];
        synMino = [synMino(:,1:(end-1)), synMino(:,end)];
    end
    %trainLOCTemp = [trainXOri(:,LOCIdx); synMino(:,LOCIdx)];
    
    %[mappedX, mapping] = kernel_pca(newTrainX, 6);
    if ~isempty(synMino)
        newTrainXBal = [newTrainX; synMino(:,1:(end-1))];
        newTrainYBal = [newTrainY; synMino(:,end)];
        
%         newTrainXBal = [matData(:,1:(end-1)); synMino(:,1:(end-1))];
%         newTrainYBal = [matData(:,end); synMino(:,end)];
    else
        newTrainXBal = newTrainX;
        newTrainYBal = newTrainY;
        
%         newTrainXBal = matData(:,1:(end-1));
%         newTrainYBal = matData(:,end);
    end
    
    
    % 
    switch learner
        case 'LR'
            mdl = fitlm(newTrainXBal, newTrainYBal);
            fitY = predict(mdl, trainXOri(:,idxSelFea)); %
        case 'CART'
            mdl = fitrtree(newTrainXBal,newTrainYBal);
            fitY = predict(mdl, trainXOri(:,idxSelFea)); %
        case 'RF'
            nTrees = 20;
            mdl = TreeBagger(nTrees, newTrainXBal, newTrainYBal, 'Method','regression');
            fitY = predict(mdl, trainXOri(:,idxSelFea)); %
        case 'BRR'
            mdl = py.sklearn.linear_model.BayesianRidge();
            mdl.fit(newTrainXBal,newTrainYBal);
            fitY = mdl.predict(trainXOri(:,idxSelFea));
            fitY = (fitY.data.double)';
        case 'GBR'
            mdl = py.sklearn.ensemble.GradientBoostingRegressor('ls', 0.1, int64(20), 1.0, 'friedman_mse', int64(2), int64(1), 0.0, int64(3), 0.0, py.None, py.None, int64(0));
            mdl.fit(newTrainXBal,newTrainYBal);
            fitY = mdl.predict(trainXOri(:,idxSelFea));
            fitY = (fitY.data.double)';
        case 'SVR'
            rand('seed',0);
            mdl = fitrsvm(newTrainXBal,newTrainYBal,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);
            fitY = predict(mdl, trainXOri(:,idxSelFea));
        case {'pr','nbin','zip'}
            mdl = py.CountModelFun.count_model(py.numpy.array([newTrainXBal,newTrainYBal]), py.numpy.array([trainXOri(:,idxSelFea),trainYOri]), learner);
            fitY = mdl{1}.double;
    end
    if useDensity % 
        fitY = round(fitY.*trainLOC); 
    else
        fitY = round(fitY); 
    end
    
    fitY(fitY<0) = 0;
    tempTr = RegPerformance(trainYOri, fitY, trainLOC);
    
%     switch learner
%         case {'pr','nbin','zip'}
%             weight(1,j0) = tempTr.fpa;
%             weight(2,j0) = tempTr.kendall;
%         otherwise
%             weight(1,j0) = tempTr.rmse;
%             weight(2,j0) = tempTr.fpa;
%     end
    
    weight(1,j0) = tempTr.rmse; % 
    weight(2,j0) = tempTr.fpa;  % 
    modelCell{j0} = mdl;
    
    % Prediction
    switch learner
        case {'BRR','GBR'}
            preDes = mdl.predict(testXOri(:,idxSelFea));
            % preNum = round((preDes.data.double)'.*testLOC);
            if useDensity
                preNum = round((preDes.data.double)'.*testLOC);
            else
                preNum = round((preDes.data.double)');
            end
        case {'pr','nbin','zip'}
            test_x = [ones(size(testXOri,1),1), testXOri(:,idxSelFea)];
            preDes = mdl{3}.predict(mdl{2}.reshape(int64(-1), int64(1)), py.numpy.array(test_x), py.numpy.array(ones(size(test_x,1),1)).reshape(int64(size(test_x,1)),int64(1)));
            %preNum = round(preDes.double);
            if useDensity
                preNum = round((preDes.double).*testLOC);
            else
                preNum = round(preDes.double);
            end
%             temp1 = RegPerformance(testYOri, preNum, testLOC);
%             mdl = py.CountModelFun.count_model(py.numpy.array([newTrainXBal,newTrainYBal]), py.numpy.array([testXOri(:,idxSelFea),testYOri]), learner);
%             preY = mdl{1}.double;
%             temp2 = RegPerformance(testYOri, preY, testLOC);
        otherwise
            preDes = predict(mdl, testXOri(:,idxSelFea));
            % preNum = round(preDes.*testLOC);
            if useDensity
                preNum = round(preDes.*testLOC);
            else
                preNum = round(preDes);
            end
    end
    
    preNum(preNum<0) = 0;
    preES(:,j0) = preNum; % 
    
    
    temp = RegPerformance(testYOri, preES(:,j0), testLOC);
    perfArra(j0,:) = [temp.rmse,  temp.kendall, temp.ptop20, temp.fpa, temp.popt, temp.popt20];
end

% 
weight(1,:) = sum(weight(1,:))./(weight(1,:)+eps); 
weight(1,:) = weight(1,:)/(sum(weight(1,:))+eps);
preE = round(preES*weight(1,:)');

% %  
% weight(2,:) = weight(2,:)/(sum(weight(2,:))+eps);
% preE = round(preES*weight(2,:)');

% weight(2,:) = weight(2,:)/(sum(weight(2,:))+eps); % 权重归一化 for FPA
% preE = round(preES*weight(2,:)');


perf = RegPerformance(testYOri, preE, testLOC);

end

