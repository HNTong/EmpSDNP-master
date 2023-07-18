function SynSamples = SMOTEND(dataSet, ideRatio, k, seed)
%SMOTEND Summary of this function goes here: Implement SMOTER algorithm.
%   Detailed explanation goes here
% INPUTS：
%   (1) dataSet - a n*(d+1) matrix;
%   (2) ideRatio - the ideal imbalance ratio after resampling; 
% OUTPUTS：
%   SynSamples: the sythetic minority class sampels.
%
% Reference: Torgo, L., et al. SMOTE for Regression. in Progress in
%       Artificial Intelligence. 2013. Berlin, Heidelberg: Springer Berlin
%       Heidelberg: Berlin, Heidelberg. p. 378-389.
%



if nargin<1
    error('Please check the parameters!');
end
if ~exist('ideRatio','var')||isempty(ideRatio)
    ideRatio = 1;
end
if ~exist('k','var')||isempty(k)
    k = 5;
end
if ~exist('seed','var')||isempty(seed)
    seed = '';
end

if ~isempty(seed) && rem(seed, 1)==0
    rng(seed);
end

n = size(dataSet,1); % Number of samples
% m = sum(dataSet(:,end)==1); % Number of minority class samples
% m = sum(~dataSet(:,end)==0); % Number of minority class samples
m = sum(dataSet(:,end)~=0); % Number of minority class samples


if m>=(n-m) % No resampling if number of defective modules is larger than number of non-defective modules. 
    SynSamples = [];
    return;
end

minoSam = dataSet(dataSet(:,end)~=0,:); % 
majoSam = dataSet(dataSet(:,end)==0,:);  % 标签为零代表多数类样本
minoSamX = minoSam(:,1:end-1);


gNum = (n-m)*ideRatio-m; % Number of synthetic minority class samples
if gNum < m % 
    indexOri = 1;
    m = floor(gNum);
else
    indexOri = floor(gNum/m); % 
end

% 每个正样本的正类近邻
D = dist(minoSamX'); % Dij表示第i个样本到第j个样本的欧式距离， D是一个对称矩阵；
D = D - eye(size(D,1),size(D,1)); % 对角线元素全部变为负数，后续好删除
[~, idx] = sort(D, 2); % D中每行元素按升序排列
idx = idx(:,2:end); % 剔除本身（第一列）；近邻索引（相对于minoSam，不是dataSet）


SynSamples = zeros(m*indexOri, size(dataSet,2)); % 初始化
count = 1; % 初始化
for i=1:m % each defective sample, m>=2
    index = indexOri;
    
    while index
        if k<=size(idx,2) % 
            nn = idx(i,randperm(k,1)); % minoSam的第i个少数类样本的前k个近邻中的任意一个近邻
        else
            
            temp0 = size(idx,2);
            temp = randperm(temp0,1);
            nn = idx(i,temp,1); % minoSam的第i个少数类样本的前k个近邻中的任意一个近邻
            
        end
        
        xnn = minoSamX(nn,:);
        xi = minoSamX(i,:);
        xSyn = xi + rand * (xnn - xi);
        d1 = norm(xSyn - xi); % distance between xSyn and xi
        d2 = norm(xSyn - xnn); % distance between xSyn and xnn
        ySyn = (d2*minoSam(i,end)+d1*minoSam(nn,end))/(d1+d2);

%         ySyn = round(ySyn);
        SynSamples(count,:) = [xSyn, ySyn];
        count = count + 1;
        index = index - 1;
    end
end

SynSamples(isnan(SynSamples(:,end)),:) = []; % when minoSam has same instances, it will result in NAN.

end


