function popt20 = Ptop20(actNum, preNum)
%PTOP20 Summary of this function goes here
%   Detailed explanation goes here
% INPUTS：
%   (1) actNum - actual value
%   (2) preNum - predicted value
% OUTPUTS：
%
%
% Reference：Weyuker E J , Ostrand T J , Bell R M . Comparing the effectiveness of several modeling methods for fault prediction. Empirical Software Engineering, 2010, 15(3):277-295.

% 
preNum = round(preNum);

% Step1: 
[~, idx] = sort(preNum); % 

% Step2: 
s_actNum = actNum(idx);

% Step3
K = length(actNum);
N = sum(actNum);
m = floor(0.2*K);
popt20 = 0;
for i=K-m+1:K
    popt20 = popt20 + s_actNum(i);
end

popt20 = popt20/N;
end

