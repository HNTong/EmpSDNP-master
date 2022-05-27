function SynSamples = SMOTEND(dataSet, ideRatio, k, seed)
%SMOTEND Summary of this function goes here: Implement SMOTER algorithm.
%   Detailed explanation goes here
% INPUTS��
%   (1) dataSet - ����������������һ��n*(d+1)����nΪ��������dΪ������Ŀ�����һ��Ϊ��ǩ��{0,1}������1��ʾ�����ࣻע�⣺���ܺ����ظ����������ͻ������������ͬ�����������ͬ��
%   (2) ideRatio - ����������������������������������ı�ֵ��
% OUTPUTS��
%   SynSamples: ���ɵ�����������
%
% Reference: Torgo, L., et al. SMOTE for Regression. in Progress in
%       Artificial Intelligence. 2013. Berlin, Heidelberg: Springer Berlin
%       Heidelberg: Berlin, Heidelberg. p. 378-389.
%
% Written by Haonan Tong, hntong@bjtu.edu.cn.


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
majoSam = dataSet(dataSet(:,end)==0,:);  % ��ǩΪ��������������
minoSamX = minoSam(:,1:end-1);


gNum = (n-m)*ideRatio-m; % Number of synthetic minority class samples
if gNum < m % 
    indexOri = 1;
    m = floor(gNum);
else
    indexOri = floor(gNum/m); % 
end

% ÿ�����������������
D = dist(minoSamX'); % Dij��ʾ��i����������j��������ŷʽ���룬 D��һ���Գƾ���
D = D - eye(size(D,1),size(D,1)); % �Խ���Ԫ��ȫ����Ϊ������������ɾ��
[~, idx] = sort(D, 2); % D��ÿ��Ԫ�ذ���������
idx = idx(:,2:end); % �޳�������һ�У������������������minoSam������dataSet��


SynSamples = zeros(m*indexOri, size(dataSet,2)); % ��ʼ��
count = 1; % ��ʼ��
for i=1:m % each defective sample, m>=2
    index = indexOri;
    
    while index
        if k<=size(idx,2) % 
            nn = idx(i,randperm(k,1)); % minoSam�ĵ�i��������������ǰk�������е�����һ������
        else
            
            temp0 = size(idx,2);
            temp = randperm(temp0,1);
            nn = idx(i,temp,1); % minoSam�ĵ�i��������������ǰk�������е�����һ������
            
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


