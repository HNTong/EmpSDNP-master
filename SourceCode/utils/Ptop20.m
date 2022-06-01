function popt20 = Ptop20(actNum, preNum)
%PTOP20 �˴���ʾ�йش˺�����ժҪ
% ���ܣ�����Popt20%
% ���������
%   (1) actNum - ʵ��ֵ
%   (2) preNum - Ԥ��ֵ
% ���������
%
%
% �ο����ף�Weyuker E J , Ostrand T J , Bell R M . Comparing the effectiveness of several modeling methods for fault prediction. Empirical Software Engineering, 2010, 15(3):277-295.

% ��������
preNum = round(preNum);

% Step1: ��Ԥ��ֵ������������
[~, idx] = sort(preNum); % 

% Step2: ������Ӧ��ʵ��ֵ��˳��
s_actNum = actNum(idx);

% Step:�������ս��
K = length(actNum);
N = sum(actNum);
m = floor(0.2*K);
popt20 = 0;
for i=K-m+1:K
    popt20 = popt20 + s_actNum(i);
end

popt20 = popt20/N;
end

