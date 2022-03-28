% This main function is related to A survey of sparse representation: algorithms and applications
% of our paper in IEEE ACCESS.
% note:
% fea                    ----      the database and each row is a sample vector
% gnd                    ----      label of the samples
% separate_data_rand     ----      Separate data into training samples and test samples
% train_num              ----      number of train for each subject
% lambda                 ----      the regularization parameter for sparse representation
% train_data             ----      the train data matrix
% train_label            ----      the label of the training samples
% test_data              ----      the test data matrix
% test_label             ----      the label of the test samples
% Ravg                   ----      average classification accuracy
% Rstd                   ----      standard deviation of the classification accuracies
% RavgT                  ----      average time-consuming

% Copyright. Zheng Zhang, HITSZ, Shenzhen
% references:
% Z. Zhang, Y. Xu, J. Yang, X. Li, D. Zhang, A survey of sparse representation: algorithms and applications,
% DOI: 10.1109/ACCESS.2015.2430359. In IEEE ACCESS, 2015.
%---------------------------------------------------------------------------------------------------------
% J. A. Tropp and A. C. Gilbert, Signal recovery from random measurements via orthogonal matching pursuit, 
% IEEE Transactions on Information Theory, vol. 53, no. 12, pp. 4655¨C4666, 2007.
% S.J. Kim, K. Koh, M. Lustig, S. Boyd, and D. Gorinevsky. A method for large-scale l1-regularized
% least squares. IEEE Journal on Selected Topics in Signal Processing, 1(4):606¨C617, 2007.
% A. Beck and M. Teboulle, A fast iterative shrinkage thresholding algorithm for linear inverse problems,
% SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183¨C202, 2009.
% M. Asif, J.Romberg. Sparse signal recovery for streaming signals using L1-homotopy,
% IEEE Transactions on  Signal Processing,  vol. 62, no. 16, pp. 4209 - 4223, 2014.
% A. Y. Yang, Z. Zhou, A. Balasubramanian, S. Sastry, Y. Ma. Fast-minimization algorithms for robust face recognition,
% IEEE Transactions onImage Processing, 22(8): 3234-3246, 2013.
% Y. Xu, D. Zhang, J. Yang, J.-Y. Yang, A two-phase test sample sparse representation method for use with face recognition, 
% IEEE Transactions on Circuits and Systems for Video Technology, 21(9), 1255-1262, 2011.

clear all
close all
clc
addpath ./solvers
addpath ./solvers/OMPbox
addpath ./databases
addpath ./utility

load 'data_ORL'
fea = double(fea);

par = [];
par.class_num = length(unique(gnd));
for train_num = 1:6
    methods = {'OMP','L1LS','PALM','FISTA','DALM','Homotopy','TPTSR'};%
    for i = 1:length(methods)
        fprintf('Method Number_train  & accuracy / time \n');
        for iter =1:10
            lambda = [0 0.02 0 0.03 0.03 0.03 0.05] ; % Tune the regularization parameters to obtain the best results
            par.lambda = lambda(i);
            [train_data, train_label, test_data, test_label] = separate_data_rand(fea,gnd,train_num);
            par.dim = length(train_label)-1; % The dimension of the samples after dimension reduction
            fprintf([methods{i}  ' ' 'train_num=' num2str(train_num) ' '])
            [accracy, time] = SRCMethod(methods{i}, train_num, train_data, train_label, test_data, test_label, par);
            acc(iter,1) = accracy;
            T(iter,1) = time;
            clearvars -except train_num m class_num row col fea gnd iter methods i par acc Ravg Rstd T lambda;
        end
        Ravg= mean(acc);
        Rstd = std(acc);    
        RavgT = mean(T);
        fprintf('\\\\ \n')
    end
end
