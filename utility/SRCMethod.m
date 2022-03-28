function [acc, time] = SRCMethod(method,train_num,train_data,train_label,test_data,test_label,par)
%
% 'OMP','GPSR','L1LS','FISTA','SpaRSA','SolveL1_2','PALM','DALM','Homotopy','TPTSR'
%
switch(method)
    case 'OMP'
        G = train_data*train_data';
        Gamma = omp(train_data*test_data',G,0.05*length(train_label));
        t0 = cputime;
        N = length(test_label);            
        for id = 1:N
            testVec=test_data(id,:);
%             fprintf('The %dth testing sample is processing...\r\n',id); 
            coeff = Gamma(:,id);
            classPredict(id,1) = classify_term(train_data,testVec,par.class_num,coeff,train_num);
        end
        % compute accuracy
        acc = 100*(sum(classPredict==test_label))/N;
        time = cputime-t0;
        fprintf('& %2.2f / %5.2f  \n',acc,time);   
        
    case 'L1LS'
        t0 = cputime;
        N = length(test_label);            
        for id = 1:N
            testVec=test_data(id,:);
%             fprintf('The %dth testing sample is processing...\r\n',id); 
            coeff = l1_ls(train_data',testVec',par.lambda);
            classPredict(id,1) = classify_term(train_data,testVec,par.class_num,coeff,train_num);
        end
        % compute accuracy
        acc = 100*(sum(classPredict==test_label))/N;
        time = cputime-t0;
        fprintf('& %2.2f / %5.2f  \n',acc,time);   
        
    case 'FISTA'
        t0 = cputime;
        N = length(test_label);            
        for id = 1:N
            testVec=test_data(id,:);
%             fprintf('The %dth testing sample is processing...\r\n',id); 
            x0 = train_data*testVec';
            maxIteration = 5000;
            isNonnegative = false;
            lambda = par.lambda; %5e-3;
            tolerance = 0.001;
            STOPPING_GROUND_TRUTH = -1;
            STOPPING_DUALITY_GAP = 1;
            STOPPING_SPARSE_SUPPORT = 2;
            STOPPING_OBJECTIVE_VALUE = 3;
            STOPPING_SUBGRADIENT = 4;
            stoppingCriterion = STOPPING_GROUND_TRUTH;
            [coeff, iterationCount] = SolveFISTA(train_data', testVec', ...
                    'maxIteration', maxIteration,...
                    'stoppingcriterion', stoppingCriterion, ...
                    'isNonnegative', isNonnegative, ... 
                    'groundtruth', x0, ...
                    'lambda', lambda, ...
                    'tolerance', tolerance); 
            classPredict(id,1) = classify_term(train_data,testVec,par.class_num,coeff,train_num);
        end
        % compute accuracy
        acc = 100*(sum(classPredict==test_label))/N;
        time = cputime-t0;
        fprintf('& %2.2f / %5.2f  \n',acc,time);   

    case 'SolveL1_2'
        t0 = cputime;
        N = length(test_label);            
        L =  norm(train_data*train_data','fro'); 
        for id = 1:N
            testVec=test_data(id,:);
%             fprintf('The %dth testing sample is processing...\r\n',id); 
            [index x it] = SolveL1_2(train_data',testVec',par,train_num,L);
            classPredict(id,1) = index;
        end
        % compute accuracy
        acc = 100*(sum(classPredict==test_label))/N;
        time = cputime-t0;
        fprintf('& %2.2f / %5.2f  \n',acc,time);   
              
    case 'PALM'
        t0 = cputime;
        N = length(test_label);            
        for id = 1:N
            testVec=test_data(id,:);
%             fprintf('The %dth testing sample is processing...\r\n',id); 
        x0 = train_data*testVec';
        maxIteration = 5000;
        tolerance = 0.001;
        STOPPING_GROUND_TRUTH = -1;
        STOPPING_DUALITY_GAP = 1;
        STOPPING_SPARSE_SUPPORT = 2;
        STOPPING_OBJECTIVE_VALUE = 3;
        STOPPING_SUBGRADIENT = 4;
        stoppingCriterion = STOPPING_GROUND_TRUTH;
        [coeff, iterationCount] = SolvePALM(train_data', testVec', ...
                'maxIteration', maxIteration,...
                'stoppingcriterion', stoppingCriterion, ...
                'groundtruth', x0, ...
                'tolerance', tolerance); 
        classPredict(id,1) = classify_term(train_data,testVec,par.class_num,coeff,train_num);
        end
        % compute accuracy
        acc = 100*(sum(classPredict==test_label))/N;
        time = cputime-t0;
        fprintf('& %2.2f / %5.2f  \n',acc,time);   
        
    case 'DALM'
        t0 = cputime;
        N = length(test_label);
        for id = 1:N
            testVec=test_data(id,:);
%             fprintf('The %dth testing sample is processing...\r\n',id); 
        x0 = train_data*testVec';
        maxIteration = 5000;
        lambda = par.lambda; %5e-3;
        tolerance = 0.001;
        STOPPING_GROUND_TRUTH = -1;
        STOPPING_DUALITY_GAP = 1;
        STOPPING_SPARSE_SUPPORT = 2;
        STOPPING_OBJECTIVE_VALUE = 3;
        STOPPING_SUBGRADIENT = 4;
        stoppingCriterion = STOPPING_GROUND_TRUTH;
        [coeff, iterationCount] = SolveDALM(train_data', testVec', ...
                'maxIteration', maxIteration,...
                'stoppingcriterion', stoppingCriterion, ...
                'groundtruth', x0, ...
                'lambda', lambda, ...
                'tolerance', tolerance); 
        classPredict(id,1) = classify_term(train_data,testVec,par.class_num,coeff,train_num);
        end
        % compute accuracy
        acc = 100*(sum(classPredict==test_label))/N;
        time = cputime-t0;
        fprintf('& %2.2f / %5.2f  \n',acc,time);   
        
    case 'Homotopy'
        t0 = cputime;
        N = length(test_label);
        for id = 1:N
            testVec=test_data(id,:);
%             fprintf('The %dth testing sample is processing...\r\n',id); 
            x0 = train_data*testVec';
            maxIteration = 5000;
            isNonnegative = false;
            lambda = par.lambda; %5e-3;
            tolerance = 0.001;
            STOPPING_GROUND_TRUTH = -1;
            STOPPING_DUALITY_GAP = 1;
            STOPPING_SPARSE_SUPPORT = 2;
            STOPPING_OBJECTIVE_VALUE = 3;
            STOPPING_SUBGRADIENT = 4;
            stoppingCriterion = STOPPING_GROUND_TRUTH;%SolveHomotopy
            [coeff, iterationCount] = SolveHomotopy(train_data', testVec', ...
                    'maxIteration', maxIteration,...
                    'stoppingcriterion', stoppingCriterion, ...
                    'isNonnegative', isNonnegative, ... 
                    'groundtruth', x0, ...
                    'lambda', lambda, ...
                    'tolerance', tolerance); 
            classPredict(id,1) = classify_term(train_data,testVec,par.class_num,coeff,train_num);
        end
        % compute accuracy
        acc = 100*(sum(classPredict==test_label))/N;
        time = cputime-t0;
        fprintf('& %2.2f / %5.2f  \n',acc,time);   
        
    case 'TPTSR'
        t0 = cputime;
        N = length(test_label);
        result = inv(train_data*train_data' + par.lambda*eye(par.class_num*train_num))*train_data;            
        classify_class = zeros(N,1);
        firstSolution = zeros(par.class_num*train_num,1);
        First_KNN = zeros(N,par.class_num*train_num);
        for id = 1:N
            testVec = test_data(id,:);
%             fprintf('The %dth testing sample is processing...\r\n',j); 
            firstSolution = result*testVec';
            contribution = zeros(par.class_num*train_num, par.dim);
            resi = zeros(par.class_num*train_num,1);
            for i = 1:par.class_num*train_num
                contribution(i,:) = firstSolution(i,1)*train_data(i,:);
                resi(i,1) = norm(testVec-contribution(i,:));
            end
            %把距离按从小到大排序
            list_index = zeros(par.class_num*train_num,1);
            [list_value list_index] = sort(resi); 
            First_KNN(id,:) = list_index;
%             [order_value order_index]=min(resi); 
%             classify_class(id)=floor((order_index-1)/train_num)+1;
        end
        n_number = floor(0.25 * length(train_label));
        for j = 1:N
            %取出测试样本
            tmpY = test_data(j,:); 
            newTrain = zeros(n_number,par.dim);
            TempOrder = zeros(n_number,1);
            TempOrder = First_KNN(j,1:n_number);
            newTrain = train_data(First_KNN(j,1:n_number),:);

            solution = inv(newTrain*newTrain'+par.lambda*eye(n_number))*newTrain*tmpY';
            contribution = zeros(par.dim, par.class_num);
            for i = 1:n_number
                temp_class = floor((TempOrder(i)-1)/train_num)+1;
                contribution(:,temp_class) = contribution(:,temp_class)+solution(i)*newTrain(i,:)';
            end
            resi = zeros(1,par.class_num);
            for kk = 1:par.class_num   
                resi(kk) = norm(tmpY'-contribution(:,kk));     
            end
            [min_value min_index] = min(resi);
            classPredict(j,1) = min_index;       
        end
        % compute accuracy
        acc = 100*(sum(classPredict==test_label))/N;
        time = cputime-t0;
        fprintf('& %2.2f / %5.2f  \n',acc,time);   
end