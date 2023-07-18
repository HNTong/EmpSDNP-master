function fpa = FPA(actY, preY)
%FPA Summary of this function goes here
%   Detailed explanation goes here
% INPUTS：
%   (1) actY - The actual number of defects in samples
%   (2) preY - The predicted number of defects in samples
%
% Reference：Weyuker E J , Ostrand T J , Bell R M . Comparing the effectiveness of several modeling methods for fault prediction. Empirical Software Engineering, 2010, 15(3):277-295.

% Round
% preY = round(preY);

% Step1: 
[~, idx] = sort(preY); % 

% Step2: 
sorted_act_target = actY(idx);

% Step3:
temp = 0;
K = length(actY);
N = sum(actY);
for k = numel(actY):-1:1
    temp = temp + k*sorted_act_target(k); 
end
fpa = temp / (K*N+eps); % eps is added in order to prevent the case that the denominator is zero when N=0.

end

