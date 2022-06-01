function outputs = RegPerformance(actY, preY, LOC)
%REGPERFORMANCE �˴���ʾ�йش˺�����ժҪ
% ���ܣ� ����ع�����
% ���������
%   (1) actY - n*1 vector
%   (2) preY - n*1 vector
% ���������
%   outputs
%

if~exist('LOC','var')||isempty(LOC)
    LOC = 0;
end

if size(actY,2)>1||size(preY,2)>1
    assert('Inputs must be column vetor');
end

% % �������� - �ڲ������ݲ�ȫΪ���ǰ���£����ģ��Ԥ��ֵȫΪ�㣬���ģ��FPA��������������������ݸպ����������У���Ԥ��ֵȫΪ���ģ�ͻ������ŵ�ģ�ͣ�FPA=1��
% [~, idx] = sort(actY,'descend');
% actY = actY(idx);
% preY = preY(idx);
% LOC = LOC(idx);


maxIte = 1;
for i=1:maxIte
    
%     % shuffle
%     idx = randperm(numel(actY), numel(actY));
%     actY = actY(idx);
%     preY = preY(idx);
%     LOC = LOC(idx);
    
    preY = round(preY);
    preY(preY<0) = 0;
    
    fpaV(i) = FPA(actY, preY);
    areV(i) = mean(abs(preY-actY)./(actY+1));
    rmseV(i) = sqrt(mean((preY-actY).^2));
    
    kendallV(i) = corr(actY, preY, 'type' , 'kendall');  % NOTE: must be column vector; kendall ranges from �C1 to +1. A value of �C1 indicates perfect negative correlation, while a value of +1 indicates perfect positive correlation. A value of 0 indicates no correlation between the columns.
    ptop20V(i) = Ptop20(actY, preY);
    
    if LOC~=0
        popt20V(i) = CalculatePopt([actY, preY, LOC]);
        poptV(i) = CalculatePopt([actY, preY, LOC],1);
    end
end
% NAN -> 0
kendallV(isnan(kendallV)) = 0; % �����޹�

outputs.fpa = nanmean(fpaV);
outputs.kendall = nanmean(kendallV);
outputs.ptop20 = nanmean(ptop20V);
outputs.rmse = nanmean(rmseV);
outputs.popt = nanmean(poptV);
outputs.popt20 = nanmean(popt20V);
outputs.are = nanmean(areV);
end

