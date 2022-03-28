function [traindata, train_label, testdata, test_label] = separate_data_rand(fea,gnd,train_num)
%% Separate samples
	%------------------------------------------------
    % Compute the number of each class
    class_num = length(unique(gnd));
    num_class = zeros(class_num,1);
    for i=1:class_num
        num_class(i,1) = length(find(gnd==i));
    end
    
    %------------------------------------------------
    % Initialize and Seperate train and test
    train_data = []; test_data = []; 
    train_label = []; test_label = [];

    for j = 1:class_num
        index = find(gnd == j); 
		randIndex = randperm(num_class(j));
        %randIndex = 1:num_class(j);
        train_data = [train_data ; fea(index(randIndex(1:train_num)),:)]; % random choose train
        train_label = [train_label ; gnd(index(randIndex(1:train_num)))];
        test_data = [test_data ; fea(index(randIndex(train_num+1:end)),:)]; % the rest for testing
        test_label = [test_label ; gnd(index(randIndex(train_num+1:end)))];
    end
    % Nomalize all the samples
    train_data = train_data./repmat(sqrt(sum(train_data.*train_data,2)),[1 size(train_data,2)]);
    test_data = test_data./repmat(sqrt(sum(test_data.*test_data,2)),[1 size(test_data,2)]);
    
    % feature extraction
    dim = length(train_label)-1;
    R = Eigenface_f(train_data',dim); %PCA
    traindata =  train_data * R;
    testdata =  test_data * R;
    traindata = traindata./repmat(sqrt(sum(traindata.*traindata,2)),[1 dim]);
    testdata = testdata./repmat(sqrt(sum(testdata.*testdata,2)),[1 dim]);
end