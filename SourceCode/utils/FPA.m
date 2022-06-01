function fpa = FPA(actY, preY)
%FPA �˴���ʾ�йش˺�����ժҪ
% ���ܣ�����FPA
% ���������
%   (1) actY - The actual number of defects in samples
%   (2) preY - The predicted number of defects in samples
%
% �ο����ף�Weyuker E J , Ostrand T J , Bell R M . Comparing the effectiveness of several modeling methods for fault prediction. Empirical Software Engineering, 2010, 15(3):277-295.

% ��������
% preY = round(preY);

% Step1: ��Ԥ��ֵ������������
[~, idx] = sort(preY); % 

% Step2: ������Ӧ��ʵ��ֵ��˳��
sorted_act_target = actY(idx);

% Step3:����
temp = 0;
K = length(actY);
N = sum(actY);
for k = numel(actY):-1:1
    temp = temp + k*sorted_act_target(k); 
end
fpa = temp / (K*N+eps); % eps is added in order to prevent the case that the denominator is zero when N=0.

end

