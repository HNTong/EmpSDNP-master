function fpa = FPA(actY, preY)
%FPA 此处显示有关此函数的摘要
% 功能：计算FPA
% 输入参数：
%   (1) actY - The actual number of defects in samples
%   (2) preY - The predicted number of defects in samples
%
% 参考文献：Weyuker E J , Ostrand T J , Bell R M . Comparing the effectiveness of several modeling methods for fault prediction. Empirical Software Engineering, 2010, 15(3):277-295.

% 四舍五入
% preY = round(preY);

% Step1: 对预测值进行升序排列
[~, idx] = sort(preY); % 

% Step2: 调整对应的实际值的顺序
sorted_act_target = actY(idx);

% Step3:计算
temp = 0;
K = length(actY);
N = sum(actY);
for k = numel(actY):-1:1
    temp = temp + k*sorted_act_target(k); 
end
fpa = temp / (K*N+eps); % eps is added in order to prevent the case that the denominator is zero when N=0.

end

