function popt20 = Ptop20(actNum, preNum)
%PTOP20 此处显示有关此函数的摘要
% 功能：计算Popt20%
% 输入参数：
%   (1) actNum - 实际值
%   (2) preNum - 预测值
% 输出参数：
%
%
% 参考文献：Weyuker E J , Ostrand T J , Bell R M . Comparing the effectiveness of several modeling methods for fault prediction. Empirical Software Engineering, 2010, 15(3):277-295.

% 四舍五入
preNum = round(preNum);

% Step1: 对预测值进行升序排列
[~, idx] = sort(preNum); % 

% Step2: 调整对应的实际值的顺序
s_actNum = actNum(idx);

% Step:计算最终结果
K = length(actNum);
N = sum(actNum);
m = floor(0.2*K);
popt20 = 0;
for i=K-m+1:K
    popt20 = popt20 + s_actNum(i);
end

popt20 = popt20/N;
end

