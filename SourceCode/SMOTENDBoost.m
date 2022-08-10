function prediction = SMOTENDBoost (TRAIN,TEST,BaseLearn,ClassDist, ErrorType)
% This function implements the SMOTENDBoost Algorithm. For more details on the 
% theoretical description of the algorithm please refer to the following 
% paper:
% Input: TRAIN = Training data, a n1*(d1+1) matrix where the last column is the number of defects {0,1,2,...}.
%        TEST = Test data, a n2*(d2+1) matrix, where the last column is the number of defects.
%        WeakLearn = String to choose algortihm. Choices are
%                    'SVR','CART','MLR' and 'RF'.
%        ClassDist = true or false. true indicates that the class
%                    distribution is maintained while doing weighted 
%                    resampling and before SMOTEND is called at each 
%                    iteration. false indicates that the class distribution
%                    is not maintained while resampling.
% Output: prediction = size(TEST,1)*2 matrix. Col 1 is class labels for 
%                      all instances. Col 2 is probability of the instances 
%                      being classified as positive class.

%%
% javaaddpath('weka.jar');
warning('off');
if ~exist('BaseLearn','var')||isempty(BaseLearn)
    BaseLearn = 'RF';
end
if ~exist('ClassDist','var')||isempty(ClassDist)
    ClassDist = true;
end
if ~exist('ErrorType','var')||isempty(ErrorType)
    ErrorType = 'square'; % 'square' 'exponent'
end

%% Training SMOTEBoost
% Total number of instances in the training set
m = size(TRAIN,1);
POS_DATA = TRAIN(~TRAIN(:,end)==0,:);
NEG_DATA = TRAIN(TRAIN(:,end)==0,:);
pos_size = size(POS_DATA,1);
neg_size = size(NEG_DATA,1);

% % Reorganize TRAIN by putting all the positive and negative exampels together, respectively.
% TRAIN = [POS_DATA;NEG_DATA];

% Total number of iterations of the boosting method
T = 50;

% W stores the weights of the instances in each row for every iteration of
% boosting. Weights for all the instances are initialized by 1/m for the
% first iteration.
W = zeros(T,m);
for i = 1:m
    W(1,i) = 1/m;
end

% L stores pseudo loss values, H stores hypothesis, B stores (1/beta) 
% values that is used as the weight of the % hypothesis while forming the 
% final hypothesis. % All of the following are of length <=T and stores 
% values for every iteration of the boosting process.
L = [];
H = {};
alpha = zeros(T, 1);

% Loop counter
t = 1;

% Keeps counts of the number of times the same boosting iteration have been repeated
count = 0;

% Boosting T iterations
while t <= T
    
    % LOG MESSAGE
    % disp (['Boosting iteration #' int2str(t)]);
    
    if ClassDist
        % Resampling POS_DATA with weights of positive example
        POS_WT = zeros(1,pos_size);
        sum_POS_WT = sum(W(t,1:pos_size));
        for i = 1:pos_size
           POS_WT(i) = W(t,i)/sum_POS_WT ;
        end
        RESAM_POS = POS_DATA(randsample(1:pos_size,pos_size,true,POS_WT),:);

        % Resampling NEG_DATA with weights of positive example
        NEG_WT = zeros(1,neg_size);
        sum_NEG_WT = sum(W(t,pos_size+1:m));
        for i = 1:neg_size
           NEG_WT(i) = W(t,pos_size+i)/sum_NEG_WT ;
        end
        RESAM_NEG = NEG_DATA(randsample(1:neg_size,neg_size,true,NEG_WT),:);
    
        % Resampled TRAIN is stored in RESAMPLED
        RESAMPLED = [RESAM_POS;RESAM_NEG];
        
        % Calulating the percentage of boosting the positive class. 'pert'
        % is used as a parameter of SMOTE
        pert = ((neg_size-pos_size)/pos_size)*100;
    else 
        % Indices of resampled train
        RND_IDX = randsample(1:m,m,true,W(t,:)); % 
        
        % Resampled TRAIN is stored in RESAMPLED
        RESAMPLED = TRAIN(RND_IDX,:);
        
        % Calulating the percentage of boosting the positive class. 'pert'
        % is used as a parameter of SMOTE
        pos_size0 = sum(RESAMPLED(:,end)~=0);
        neg_size0 = sum(RESAMPLED(:,end)==0);
        pert = ((neg_size0-pos_size0)/pos_size0)*100;
    end
    
    % 少数类样本不能太少，否则回到初始状态
    if (sum(RESAMPLED(:,end)~=0)/pos_size < 0.1)||((sum(RESAMPLED(:,end)~=0)/pos_size >= 0.1)&&sum(RESAMPLED(:,end)~=0)<2)
        % disp('hello');
        RESAMPLED = TRAIN;
        W(t,:) = 1/m;
    end
    
    if unique(W(t,:))==1/m  % t=1 or RESAMPLED被还原
        RESAMPLED = TRAIN;
    end
    

    synMinoSamples = SMOTEND(RESAMPLED);
    S = [RESAMPLED;synMinoSamples];
    
    
    
    % Training a base learner. 'pred' is the weak hypothesis. However, the hypothesis function is encoded in 'model'.
    switch BaseLearn
        case 'SVR'
            model = fitrsvm(S(:,1:(end-1)), S(:,end));
        case 'CART'
            rand('seed',0);
            model = fitrtree(S(:,1:(end-1)), S(:,end));
        case 'MLR'
            rand('seed',0);
            model = fitlm(S(:,1:(end-1)), S(:,end));
        case 'RF'
            nTrees = 100;
            rand('seed',0);
            model = TreeBagger(nTrees, S(:,1:(end-1)), S(:,end), 'Method', 'Regression');
            
            
%             testX = TEST(:,1:end-2);
%             testLOC = TEST(:,end-1);
%             testYOri = TEST(:,end);
%             preRFdes = predict(model, testX);
%             preRF = round(preRFdes.*testLOC);
%             temp = RegPerformance(testYOri, preRF, testLOC);
%             perf_RF_folds=[temp.rmse,  temp.kendall, temp.ptop20, temp.fpa];
    end
    fit = predict(model, TRAIN(:,1:(end-1)));
    
    
    % Computing the pseudo loss of hypothesis 'model'
    maxErr = max(abs(fit-TRAIN(:,end)));
    switch ErrorType
        case 'linear'
            e = abs(fit-TRAIN(:,end))./maxErr;
        case 'square'
            e = ((fit-TRAIN(:,end))/maxErr).^2; % 
            
        case 'exponent'
            e = 1-(exp(-abs(fit-TRAIN(:,end))/maxErr));
        case 'FPA'
            e = 1;
    end
    %loss = sum(e.*W(t,:)');
    loss = 1 - FPA(TRAIN(:,end).*TRAIN(:,11), fit.*TRAIN(:,11));
    
   
    
%     % If count exceeds a pre-defined threshold (5 in the current
%     % implementation), the loop is broken and rolled back to the state
%     % where loss > 0.5 was not encountered.
%     if count > 5
%        L = L(1:t-1);
%        H = H(1:t-1);
%        B = B(1:t-1);
%        disp ('          Too many iterations have loss > 0.5');
%        disp ('          Aborting boosting...');
%        break;
%     end
%     
%     % If the loss is greater than 1/2, it means that an inverted
%     % hypothesis would perform better. In such cases, do not take that
%     % hypothesis into consideration and repeat the same iteration. 'count'
%     % keeps counts of the number of times the same boosting iteration have
%     % been repeated
%     if loss > 0.5
%         count = count + 1;
%         continue;
%     else
%         count = 1;
%     end        
    
    
    L(t) = loss; % Pseudo-loss at each iteration
    H{t} = model; % Hypothesis function   
    
    alpha(t) = loss/(1-loss);

    
    % At the final iteration there is no need to update the weights any further
    if t==T
        break;
    end
    
    % Updating weight    
    W(t+1,:) = W(t,:).*(alpha(t).^(1-e))'; % row.*row
    
    % Normalizing the weight for the next iteration
    sum_W = sum(W(t+1,:));
    W(t+1,:) = W(t+1,:)./sum_W;
%     for i = 1:m
%         W(t+1,i) = W(t+1,i)/sum_W;
%     end
    
    % Incrementing loop counter
    t = t + 1;
end

% The final hypothesis is calculated and tested on the test set simulteneously.

%% Testing SMOTENDBoost

% % Case1: AdaBoost.R2
% temp = log(1/(alpha+eps));
% indFlag = temp==median(temp);
% model = H{indFlag};
% prediction = predict(model, TEST(:,1:(end-1)));


% Case2:
% Normalizing B
sum_alpha = sum(alpha);
alpha = alpha./sum_alpha;


p = zeros(size(TEST,1), numel(H));
for j = 1:numel(H) %
    p(:,j) = predict(H{j}, TEST(:,1:(end-1)));
end
prediction = p*alpha;



end