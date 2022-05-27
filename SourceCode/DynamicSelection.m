function perf = DynamicSelection(train, test, testLOC, K)
%DYNALICSELECTION Summary of this function goes here: Implement Dynamic Selection of Learning Techniques.
%   Detailed explanation goes here
% INPUTS:
%   (1) train - a n1*(d+1) matrix where the last column is the independent variable (i.e., number of defects).
%   (2) test  -  a n2*(d+1) matrix.
%   (3) K     - number of partations of validation set
% Reference: Rathore, S.S. and S. Kumar, An Approach for the Prediction of
%      Number of Software Faults Based on the Dynamic Selection of Learning
%      Techniques. IEEE Transactions on Reliability, 2018. DOI:
%      10.1109/TR.2018.2864206.
%
% Written by Haonan Tong, hntong@bjtu.edu.cn.


if ~exist('K','var')||isempty(K)
    K = 4;
end

% Initialization
L = {'DTR', 'LR', 'MLP'};
AveErr = zeros(numel(L), K);
Cntrd = [];

% Shuffle train set
n1 = size(train, 1);
idx = randperm(n1, n1);
train = train(idx, :);

% Generate validation set
idxValid = randperm(n1, floor(n1*1/4));
validation = train(idxValid, :);

% Update train set
train = train(setdiff(1:n1, idxValid),:); 
n1 = size(train, 1);

%% Train each base learner
javaaddpath('weka.jar');
wekaClassifiers = cell(1,numel(L));
for j =1:numel(L) % each base learner
    idxBootstrap = randsample(n1, n1, true); % sample uniformly at random, with replacement
    bootstrapData = train(idxBootstrap, :);
    while sum(bootstrapData(:,end)>0)<2
        idxBootstrap = randsample(n1, n1, true); % sample uniformly at random, with replacement
        bootstrapData = train(idxBootstrap, :);
    end
%     save('bootstrapData.mat','bootstrapData');
%     Rpath = 'D:\Program Files\R\R-4.0.3\bin';
%     RscriptFileName = 'E:\Document\Programming\MATLAB\Count model-v2.0\SMOTER.R';
%     RunRcode(RscriptFileName, Rpath);
%     posiSynthetic = load('synData.mat');
    posiSynthetic = SMOTEND(bootstrapData);  % Implement SMOTER method.
    if size(posiSynthetic,1)>0
        posiSynthetic(:,end) = round(posiSynthetic(:,end));
    end
    balBootstrapData = [bootstrapData; posiSynthetic];
    
    wekaDataTr = mat2ARFF(balBootstrapData);
%     wekaDataTr = mat2ARFF(bootstrapData);
    switch L{j}
        case 'DTR'
%             javaaddpath('simpleCART.jar');
            wekaClassifier = javaObject('weka.classifiers.trees.REPTree');
        case 'LR'
            wekaClassifier = javaObject('weka.classifiers.functions.LinearRegression');  
        case 'MLP'
            wekaClassifier = javaObject('weka.classifiers.functions.MultilayerPerceptron'); 
    end
    wekaClassifier.buildClassifier(wekaDataTr); 
    wekaClassifiers{j} = wekaClassifier;
end


%% Calculate AveErr
[idx, Cntrd] = kmeans(validation, K); % MATLAB build-in function. Return an n-by-1 vector (idx) containing cluster indices of each observation, K cluster centroid locations in the K-by-d matrix C.
for k=1:K % each partition of validation set
    subValid = validation(idx==k,:);
    subValidWEKA = mat2ARFF(subValid);
    for j=1:numel(L) % each base learner
        predDef = zeros(subValidWEKA.numInstances, 1);
        wekaClassifier = wekaClassifiers{j};
        for i=0:subValidWEKA.numInstances -1
            predDef(i+1,:) = (wekaClassifier.distributionForInstance(subValidWEKA.instance(i)))';
        end
        predDef(predDef<0) = 0;
        predDef = round(predDef);
        AveErr(j,k) = 1/(subValidWEKA.numInstances)*sum(abs(subValid(:,end)-predDef)); % Average Absolute Error
    end
end

%% Prediction on testing dataset
testWEKA = mat2ARFF(test);
predDef = zeros(size(test,1),1);
for i=0:testWEKA.numInstances -1
    
    [~, idx] = pdist2(Cntrd(:,1:end-1), test(i+1,1:end-1), 'euclidean', 'Smallest', 1); % Index in WEKA starts from 0, but index in MATLAB from 1.
    bestLeanerIdx = find(AveErr(:,idx)==min(AveErr(:,idx)));
    wekaClassifier = wekaClassifiers{bestLeanerIdx};
    predDef(i+1,:) = wekaClassifier.distributionForInstance(testWEKA.instance(i));
end

perf = RegPerformance(test(:,end), predDef, testLOC);

end

function arff = mat2ARFF(data, type)
% Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) data - a n*(d+1) matrix where the last column is independent variable.
%   (2) type - a string
% OUTPUTS:
%   arff     - an ARFF file

if ~exist('type','var')||isempty(type)
    type = 'regression';
end
label = cell(size(data,1),1);
if strcmp(type, 'classification')
    temp = data(:,end);
    for j=1:size(data,1)
        if (temp(j)==1)
            label{j} = 'true';
        else
            label{j} = 'false';
        end
    end %{0,1}--> {false, true}
else 
    label = num2cell(data(:,end));
end
featureNames = cell(size(data,2),1);
for j=1:(size(data,2)-1)
    featureNames{j} = ['X', num2str(j)];
end
featureNames{size(data,2)} = 'Defect';
arff = matlab2weka('data', featureNames, [num2cell(data(:,1:end-1)), label]);
end


function wekaOBJ = matlab2weka(name, featureNames, data,targetIndex)
% Convert matlab data to a weka java Instances object for use by weka
% classes. 
%
% name           - A string, naming the data/relation
%
% featureNames   - A cell array of d strings, naming each feature/attribute
%
% data           - An n-by-d matrix with n, d-featured examples or a cell
%                  array of the same dimensions if string values are
%                  present. You cannot mix numeric and string values within
%                  the same column. 
%
% wekaOBJ        - Returns a java object of type weka.core.Instances
%
% targetIndex    - The column index in data of the target/output feature.
%                  If not specified, the last column is used by default.
%                  Use the matlab convention of indexing from 1.
%
% Written by Matthew Dunham

    % if(~wekaPathCheck),wekaOBJ = []; return,end
    if(nargin < 4)
        targetIndex = numel(featureNames); %will compensate for 0-based indexing later
    end

    import weka.core.*;%£¡£¡£¡
    vec = FastVector();
    if(iscell(data))
        for i=1:numel(featureNames)
            if(ischar(data{1,i}))
                attvals = unique(data(:,i));
                values = FastVector();
                for j=1:numel(attvals)
                   values.addElement(attvals{j});
                end
                vec.addElement(Attribute(featureNames{i},values));
            else
                vec.addElement(Attribute(featureNames{i})); 
            end
        end 
    else
        for i=1:numel(featureNames)
            vec.addElement(Attribute(featureNames{i})); 
        end
    end
    wekaOBJ = Instances(name,vec,size(data,1));
    if(iscell(data))
        for i=1:size(data,1)
            inst = DenseInstance(numel(featureNames)); % DenseInstance
            for j=0:numel(featureNames)-1
               inst.setDataset(wekaOBJ);
               inst.setValue(j,data{i,j+1});
            end
            wekaOBJ.add(inst);
        end
    else
        for i=1:size(data,1)
            wekaOBJ.add(Instance(1,data(i,:)));
        end
    end
    wekaOBJ.setClassIndex(targetIndex-1);
end


function [mdata,featureNames,targetNDX,stringVals,relationName] =...
                                                weka2matlab(wekaOBJ,mode)
% Convert weka data, stored in a java weka Instances object to a matlab
% data type, (type depending on the optional mode, [] | {} )
% 
% wekaOBJ       - a java weka Instances object storing the data.
%
% mode          - optional, [] | {} (default = []) If [], returned mdata is 
%                 a numeric array and any strings in wekaOBJ are converted 
%                 to their enumerated indices. If {}, mdata is returned as 
%                 a cell array, preserving any present strings. 
%
% mdata         - an n-by-d matlab numeric or cell array, holding the data, 
%                 (n, d-featured examples). Type depends on the mode
%                 parameter. 
%
% featureNames - a cell array listing the names of each feature/attribute
%                in the same order as they appear, column-wise, in mdata.
%
% targetNDX    - the column index of the target, (output) feature/class
%                (w.r.t 1-based indexing)
%
% stringVals   - some weka features may be non-numeric. These are
%                automatically enumerated and the enumerated indices
%                returned in mdata instead of the string versions, (unless
%                in cell mode). The corresponding strings are returned in
%                stringVals, a cell array of cell arrays. Enumeration
%                begins at 0. 
%
% relationName - The string name of the relation.

% if(~wekaPathCheck),mdata = []; return,end
% if(nargin < 2)
%     mode = [];
% end

if(not(isjava(wekaOBJ)))
    fprintf('Requires a java weka object as input.');
    return;
end

mdata = zeros(wekaOBJ.numInstances,wekaOBJ.numAttributes);
for i=0:wekaOBJ.numInstances-1
    mdata(i+1,:) = (wekaOBJ.instance(i).toDoubleArray)';
end

targetNDX = wekaOBJ.classIndex + 1;

featureNames = cell(1,wekaOBJ.numAttributes);
stringVals = cell(1,wekaOBJ.numAttributes);
for i=0:wekaOBJ.numAttributes-1
    featureNames{1,i+1} = char(wekaOBJ.attribute(i).name);
    
    attribute = wekaOBJ.attribute(i);
    nvals = attribute.numValues;
    vals = cell(nvals,1);
    for j=0:nvals-1
        vals{j+1,1} = char(attribute.value(j));
    end
    stringVals{1,i+1} = vals;    
end

relationName = char(wekaOBJ.relationName);

if(iscell(mode))
   celldata = num2cell(mdata);
   for i=1:numel(stringVals)
      vals = stringVals{1,i};
      if(not(isempty(vals)))
        celldata(:,i) = vals(mdata(:,i)+1)';
      end
   end
   mdata = celldata;
    
end
end
