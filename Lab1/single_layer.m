% Author: Alice

% Single-layer perceptron

%_______________________________3.1.1______________________________________ 
%% Generation of linearly-separable data
clc 
clear all 
close all

% Parameters
%format long;
n = 100;
dim_outputs = 1;
mA = [ 2.0, 0.5]; sigmaA = 0.5;
mB = [-1.0, -0.1]; sigmaB = 0.5;
 
% Data
classA(1,:) = randn(1,n).*sigmaA + mA(1);
classA(2,:) = randn(1,n).*sigmaA + mA(2);
classB(1,:) = randn(1,n).*sigmaB + mB(1);
classB(2,:) = randn(1,n).*sigmaB + mB(2);
targetsA = ones(1,n);
targetsB = ones(1,n)*-1;
both_classes = [classA classB];
both_targets = [targetsA targetsB];

% Shuffle
N = size(both_targets,2);
indices = randperm(N);
shuffled_classes = both_classes(:,indices);
shuffled_targets = both_targets(:,indices);

% Plot 
figure
hold on;
axis([-3 3 -3 3]);
plot(classA(1,:), classA(2,:), 'b*');
plot(classB(1,:), classB(2,:), 'r*');


%_______________________________3.1.2______________________________________ 
% Classification with a single-layer perceptron and analysis
% Set parameters

% Add bias 
bias = ones(1,N);
X = [shuffled_classes; bias];
X_noBias = shuffled_classes;
T = shuffled_targets;
X_dim = size(X,1);
T_dim = dim_outputs;

% Initialize W
mean = 0;
std = 1/(N^(1/2));
W_start = randn(T_dim, X_dim).*std + mean;
W_start_no_bias = W_start(:,1:end-1);

% Parameters
conv_criteria = 0.025;
check_conv_points = 20;
eta = 0.003;   % learning rate
learning_rates = eta% [0.009 0.003 0.0001];
nr_epochs = 50;
x_1 = linspace(-5,5,10);


% Iterate max nr_epochs times

% Delta Rule
all_errAB_D = zeros(length(learning_rates), nr_epochs);
for i = 1:length(learning_rates)
    rate = learning_rates(i);
    W_old_D = W_start;
    for epoch = 1:nr_epochs
       delta_W_D = getDeltaW(W_old_D,rate, X, T);
       W_D = W_old_D + delta_W_D;
       W_old_D = W_D; 

       % Decision boundary Delta (animated)
       %x_2_D = ((-W_D(1)*x_1)-W_D(3))/W_D(2); 
       %plot(x_1,x_2_D,'g-')
       %drawnow
       

       % Add error rate
       sum_err = getError(W_old_D, X, T);
       sum_err = sum_err/(2*n);
       all_errAB_D(i, epoch) = sum_err;
    end
    
    % Check convergence
    current_errors = all_errAB_D(i,:);
    idx = find(current_errors < conv_criteria);
    if isempty(idx)
        disp('Delta algo. do not converge') 
    end
    if length(idx) < check_conv_points
        check_convergence_D = idx(1:length(idx))
    else 
        check_convergence_D = idx(1:check_conv_points)        
    end
    
    if rate == eta
        WD_final = W_D;
    end
    
end


% Perceptron 
all_err_P = zeros(length(learning_rates), nr_epochs);
for i = 1:length(learning_rates)
    rate = learning_rates(i);
    W_old_P = W_start;
    for epoch = 1:nr_epochs
       delta_W_P = getPercepW(W_old_P, rate, X, T);
       W_P = W_old_P + delta_W_P;
       W_old_P = W_P; 

       % Decision boundary Percep (animated)
       %x_2P = ((-W_P(1)*x_1)-W_P(3))/W_P(2); 
       %plot(x_1,x_2P,'k-')
       %drawnow

       % Add error rate
       sum_err = getError(W_old_P, X, T);
       all_err_P(i, epoch) = sum_err/(2*n);
    end
   
    % Check convergence
    current_errors = all_err_P(i,:);
    idx = find(current_errors < conv_criteria);
    if isempty(idx)
        disp('Perceptron algo. do not converge') 
    end
    if length(idx) < check_conv_points
        check_convergence_P = idx(1:length(idx))
    else 
        check_convergence_P = idx(1:check_conv_points)        
    end
    
    if rate == eta
        WP_final = W_P;
    end
end



% Delta Rule without bias
all_err_noBias = zeros(length(learning_rates), nr_epochs);
for i = 1:length(learning_rates)
    W_old_noBias = W_start_no_bias;
    rate = learning_rates(i);
    for epoch = 1:nr_epochs
       delta_W_noBias = getDeltaW(W_old_noBias,eta, X_noBias, T);
       W_noBias = W_old_noBias + delta_W_noBias;
       W_old_noBias = W_noBias;   

       % Decision boundary Delta without Bias (animated)
       %x_2_noBias = (-W_old_noBias(1)*x_1)/W_old_noBias(2); 
       %plot(x_1,x_2_noBias,'m-')
       %title('Decision Boundary for Delta-Rule Without Bias')
       %xlabel('x_1')
       %ylabel('x_2')
       %drawnow

       % Add error rate
       sum_err = getError(W_old_noBias, X_noBias, T);
       all_err_noBias(i, epoch) = sum_err/(2*n);
    end   
    
    % Check convergence
    current_errors = all_err_noBias(i,:);
    idx = find(current_errors < conv_criteria);
    if isempty(idx)
        disp('Delta-no Bias algo. do not converge') 
    end
    if length(idx) < check_conv_points
        check_convergence_noBias = idx(1:length(idx))  
    else 
        check_convergence_noBias = idx(1:check_conv_points)        
    end
    
    if rate == eta
        WnoBias_final = W_noBias;
    end
end




% Sequential 
all_err_seq = zeros(length(learning_rates), nr_epochs);
for i = 1:length(learning_rates)
    rate = learning_rates(i);
    W_old_i = W_start;
    for epoch = 1:nr_epochs
       for i=1:2*n
           X_i = X(:,i);
           T_i = T(:,i);
           delta_W_i = getDeltaW(W_old_i, rate, X_i, T_i);
           W_i = W_old_i + delta_W_i;
           W_old_i = W_i;
       end
       
       W_old_seq = W_old_i;
       

       % Decision boundary sequential (animated)
       %x_2_seq = ((-W_old_seq(1)*x_1)-W_old_seq(3))/W_old_seq(2); 
       %plot(x_1,x_2_seq,'b-')
       %drawnow


       % Add error rate
       sum_err = getError(W_old_seq, X, T);
       all_err_seq(i, epoch) = sum_err/(2*n);
    end
    %{
    % Check convergence
    current_errors = all_err_P(i,:);
    idx = find(current_errors < conv_criteria);
    if isempty(idx)
        disp('Delta sequential algo. do not converge') 
    end
    if length(idx) < check_conv_points
        check_convergence_seq = idx(1:length(idx)) 
    else 
        check_convergence_seq = idx(1:check_conv_points)        
    end
    %}
    if rate == eta
        Wseq_final = W_old_seq;
    end
end


% Plot final decision boundary
% Final decision boundary Delta
x_2D = ((-WD_final(1)*x_1)-WD_final(3))/WD_final(2);
plot(x_1,x_2D,'g-')

% Final decision boundary Perceptron
x_2P = ((-WP_final(1)*x_1)-WP_final(3))/WP_final(2);
plot(x_1,x_2P,'k-')

% Final decision boundary Delta without bias
x_2D_noBias = (-WnoBias_final(1)*x_1)/WnoBias_final(2); 
plot(x_1,x_2D_noBias,'m-')

% Final decision boundary Sequential
x_2D_seq = ((-Wseq_final(1)*x_1)-Wseq_final(3))/Wseq_final(2);
plot(x_1,x_2D_seq,'b-')

title('Decision Boundaries')
legend('Class A', 'Class B', 'Delta', 'Perceptron', 'No bias','Sequential')
xlabel('x_1 values') % x-axis label
ylabel('x_2 values') % y-axis label

%__________________________________________________________________________
% Plot comparisions between learning rates 
all_err1 = all_errAB_D;
all_err2 = all_err_seq;

figure 
axis([0 nr_epochs 0 0.2])
for i = 1:length(learning_rates)
    hold on
    curve1 = all_err1(i,:);
    curve2 = all_err2(i,:);
    plot(1:length(curve1), curve1)
    plot(1:length(curve2), curve2)
end
%{
legend('Delta: \eta = 0.009', 'Perceptron: \eta = 0.009','Delta: \eta = 0.003',...
    'Perceptron: \eta = 0.003','Delta: \eta = 0.0001','Perceptron: \eta = 0.0001')
xlabel('Epochs') % x-axis label
ylabel('Error rate') % y-axis label
title('Covergence of the Delta & Perceptron Algorithms')
%}
title('Covergence of the Sequential Delta & Batch Delta Algorithms')
legend('Delta (batch): \eta = 0.009','Delta (sequential): \eta = 0.009',...
    'Delta (batch): \eta = 0.003','Delta (sequential): \eta = 0.003', ...
    'Delta (batch): \eta = 0.0001', 'Delta (sequential): \eta = 0.0001')
xlabel('Epochs') % x-axis label
ylabel('Error rate') % y-axis label   


%_______________________________3.1.3______________________________________ 
%% Classification of samples that are not linearly separable
% PART 1
clc 
clear all 
close all

% Generate data
n = 100;
dim_outputs = 1;
mA = [ 1.0, 0.5]; sigmaA = 0.6;
mB = [-1.0, -0.1]; sigmaB = 0.6;

% Data
classA(1,:) = randn(1,n).*sigmaA + mA(1);
classA(2,:) = randn(1,n).*sigmaA + mA(2);
classB(1,:) = randn(1,n).*sigmaB + mB(1);
classB(2,:) = randn(1,n).*sigmaB + mB(2);
targetsA = ones(1,n);
targetsB = ones(1,n)*-1;
both_classes = [classA classB];
both_targets = [targetsA targetsB];

% Shuffle
N = size(both_targets,2);
indices = randperm(N);
shuffled_classes = both_classes(:,indices);
shuffled_targets = both_targets(:,indices);

% Plot 
figure
hold on;
axis([-3 3 -3 3]);
plot(classA(1,:), classA(2,:), 'b*');
plot(classB(1,:), classB(2,:), 'r*');

% Add bias 
bias = ones(1,N);
X = [shuffled_classes; bias];
T = shuffled_targets;
X_dim = size(X,1);
T_dim = dim_outputs;

% Initialize W
mean = 0;
std = 1/(N^(1/2));
W_start = randn(T_dim, X_dim).*std + mean;
%__________________________________________________________________________

% Parameters
conv_criteria = 0.025;
check_conv_points = 20;
eta = 0.003;   % learning rate
learning_rates = [0.009 0.003 0.0001];
nr_epochs = 100;
x_1 = linspace(-5,5,10);


% Iterate max nr_epochs times

% Delta Rule
all_errAB_D = zeros(length(learning_rates), nr_epochs);
for i = 1:length(learning_rates)
    rate = learning_rates(i);
    W_old_D = W_start;
    for epoch = 1:nr_epochs
       delta_W_D = getDeltaW(W_old_D,rate, X, T);
       W_D = W_old_D + delta_W_D;
       W_old_D = W_D; 

       % Decision boundary Delta (animated)
       %x_2_D = ((-W_D(1)*x_1)-W_D(3))/W_D(2); 
       %plot(x_1,x_2_D,'g-')
       %drawnow
       

       % Add error rate
       sum_err = getError(W_old_D, X, T);
       sum_err = sum_err/(2*n);
       all_errAB_D(i, epoch) = sum_err;
    end
    
    % Check convergence
    current_errors = all_errAB_D(i,:);
    idx = find(current_errors < conv_criteria);
    if isempty(idx)
        disp('Delta algo. do not converge') 
    end
    if length(idx) < check_conv_points
        check_convergence_D = idx(1:length(idx))
    else 
        check_convergence_D = idx(1:check_conv_points)        
    end
    
    if rate == eta
        WD_final = W_D;
    end
    
end


% Perceptron 
all_err_P = zeros(length(learning_rates), nr_epochs);
for i = 1:length(learning_rates)
    rate = learning_rates(i);
    W_old_P = W_start;
    for epoch = 1:nr_epochs
       delta_W_P = getPercepW(W_old_P, rate, X, T);
       W_P = W_old_P + delta_W_P;
       W_old_P = W_P; 

       % Decision boundary Percep (animated)
       %x_2P = ((-W_P(1)*x_1)-W_P(3))/W_P(2); 
       %plot(x_1,x_2P,'k-')
       %drawnow

       % Add error rate
       sum_err = getError(W_old_P, X, T);
       all_err_P(i, epoch) = sum_err/(2*n);
    end
   
    % Check convergence
    current_errors = all_err_P(i,:);
    idx = find(current_errors < conv_criteria);
    if isempty(idx)
        disp('Perceptron algo. do not converge') 
    end
    if length(idx) < check_conv_points
        check_convergence_P = idx(1:length(idx))
    else 
        check_convergence_P = idx(1:check_conv_points)        
    end
    
    if rate == eta
        WP_final = W_P;
    end
end


% Plot final decision boundary
% Final decision boundary Delta
x_2D = ((-WD_final(1)*x_1)-WD_final(3))/WD_final(2);
plot(x_1,x_2D,'g-')

% Final decision boundary Perceptron
x_2P = ((-WP_final(1)*x_1)-WP_final(3))/WP_final(2);
plot(x_1,x_2P,'k-')

% Plot comparisions between learning rates 
all_err1 = all_errAB_D;
all_err2 = all_err_P;
size(all_err2)
figure 
axis([0 nr_epochs 0 0.2])
for i = 1:length(learning_rates)
    hold on
    curve1 = all_err1(i,:);
    curve2 = all_err2(i,:);
    plot(1:length(curve1), curve1)
    plot(1:length(curve2), curve2)
end

legend('Delta: \eta = 0.009', 'Perceptron: \eta = 0.009','Delta: \eta = 0.003',...
    'Perceptron: \eta = 0.003','Delta: \eta = 0.0001','Perceptron: \eta = 0.0001')
xlabel('Epochs') % x-axis label
ylabel('Error rate') % y-axis label
title('Covergence of the Delta & Perceptron Algorithms for Not Linearly Separable Data')



%% PART 2  
clc 
clear all 
close all

% Generate different version of linearly non-separable data
dim_outputs = 1;
ndata = 100;
mA = [ 1.0, 0.3];    sigmaA = 0.2;
mB = [ 0.0, -0.1];   sigmaB = 0.3;
classA(1,:) = [ randn(1,round(0.5*ndata)) .* sigmaA - mA(1), ...
randn(1,round(0.5*ndata)) .* sigmaA + mA(1)];
classA(2,:) = randn(1,ndata) .* sigmaA + mA(2);
classB(1,:) = randn(1,ndata) .* sigmaB + mB(1);
classB(2,:) = randn(1,ndata) .* sigmaB + mB(2);
targetsA = ones(1,ndata);
targetsB = ones(1,ndata)*-1;
dataA = [classA; targetsA];
dataB = [classB; targetsB];

% Shuffle both classes separately 
dataA_shuffled = dataA(:,randperm(ndata));
dataB_shuffled = dataB(:,randperm(ndata));


% DIFFERENT SUBSAMPLES 
% Choose amount of data to remove from each class (i.e amount of test data)
percentageA = 0; 
percentageB = 50;
A_train = dataA_shuffled(:,1:(ndata-percentageA));
A_test = dataA_shuffled(:,(ndata-percentageA+1):ndata);
B_train = dataB_shuffled(:,1:(ndata-percentageB));
B_test = dataB_shuffled(:,(ndata-percentageB+1):ndata);
A_25_50 = A_train(1:2,:);
B_25_50 = B_train(1:2,:);

data_train = [A_train, B_train];
data_test = [A_test, B_test];
data_train_shuffled = data_train(:,randperm(size(data_train,2)));
data_test_shuffled = data_test(:,randperm(size(data_test,2)));
X_train = [data_train_shuffled(1:2,:); ones(1,size(data_train,2))]; % +bias
T_train = data_train_shuffled(3,:);
X_test = [data_test_shuffled(1:2,:); ones(1,size(data_test,2))];    % +bias
T_test = data_test_shuffled(3,:);


% Keep 80% from a subset of classA for which classA(1,:)<0 and 20% from a 
% subset of classA for which classA(1,:)>0
% 50 first of classA < 0, 50 last > 0
A_lessThan0 = dataA(:,1:size(dataA,2)/2);
idx_less = 1:size(A_lessThan0,2);
mix1_idx = idx_less(:,randperm(length(idx_less)));
stop_lessThan0 = 0.8*length(mix1_idx);
idx1_train = mix1_idx(:,1:stop_lessThan0);
idx1_test = mix1_idx(:,stop_lessThan0+1:end);
A_train_lessThan0 = A_lessThan0(:,idx1_train);
A_test_lessThan0 = A_lessThan0(:,idx1_test);

A_greaterThan0 = dataA(:,size(dataA,2)/2+1:end);
idx_greater = 1:size(A_greaterThan0,2);
mix2_idx = idx_greater(:,randperm(length(idx_greater)));
stop_greaterThan0 = 0.2*length(mix1_idx);
idx2_train = mix2_idx(:,1:stop_greaterThan0);
idx2_test = mix2_idx(:,stop_greaterThan0+1:end);
A_train_greaterThan0 = A_greaterThan0(:,idx2_train);
A_test_greaterThan0 = A_greaterThan0(:,idx2_test);

dataA_20_80_train = [A_train_lessThan0 A_train_greaterThan0];
dataA_20_80_test = [A_test_lessThan0 A_test_greaterThan0];
A_20_80 = dataA_20_80_train(1:2,:);
B_20_80 = classB;
data_train_20_80 = [dataA_20_80_train dataB];
data_train_20_80_shuffled = data_train_20_80(:,randperm(size(data_train_20_80,2)));
X_20_80_train = [data_train_20_80_shuffled(1:2,:); ones(1,size(data_train_20_80,2))];
T_20_80_train = [data_train_20_80_shuffled(3,:)];
data_test_20_80_shuffled = dataA_20_80_test(:,randperm(size(dataA_20_80_test,2)));
X_20_80_test = [data_test_20_80_shuffled(1:2,:); ones(1,size(dataA_20_80_test,2))];
T_20_80_test = [data_test_20_80_shuffled(3,:)];


% Choose which data to use
X = X_20_80_train;%X_train;%
T = T_20_80_train;%T_train;%
X_test = X_20_80_test; 
T_test = T_20_80_test;
A_test = dataA_20_80_test;
B_test = 0;
pointsA = A_20_80;%A_25_50;%
pointsB = B_20_80;%B_25_50;%



% Plot
figure
hold on;
axis([-2 2 -2 2]);
plot(pointsA(1,:), pointsA(2,:), 'b*');
plot(pointsB(1,:), pointsB(2,:), 'r*');


% Parameters
conv_criteria = 0.025;
check_conv_points = 20;
eta = 0.003;   % learning rate
rate = eta;
max_iters = 10;
nr_epochs = 50;
x_1 = linspace(-5,5,10);
ndata_A = size(pointsA,2);
ndata_B = size(pointsB,2);


% Delta Rule

allTest_errA = zeros(max_iters,nr_epochs);
allTest_errB = zeros(max_iters,nr_epochs);
mean_train_errors = zeros(2,max_iters);
mean_test_errors = zeros(2,max_iters);
for i = 1:max_iters
    
    % Initialize W
    X_dim = size(X,1);
    T_dim = dim_outputs;
    N = size(X,2)
    mean = 0;
    std = 1/(N^(1/2));
    W_start = randn(T_dim, X_dim).*std + mean;
    all_errAB_D = zeros(2, nr_epochs);
    W_old_D = W_start;
    for epoch = 1:nr_epochs
       delta_W_D = getDeltaW(W_old_D,rate, X, T);
       W_D = W_old_D + delta_W_D;
       W_old_D = W_D; 
       %{
       % Decision boundary Delta (animated)
       x_2_D = ((-W_D(1)*x_1)-W_D(3))/W_D(2); 
       plot(x_1,x_2_D,'g-')
       title('Decision Boundaries for 50 Epochs for the 20-80-Dataset')
       xlabel('x_1')
       ylabel('x_2')
       drawnow
       %}

       % Add error rate
       [sum_errA, sum_errB] = getError2(W_old_D, X, T);
       err_ratioA = sum_errA/(ndata_A);
       err_ratioB = sum_errB/(ndata_B);
       all_errAB_D(1, epoch) = err_ratioA;
       all_errAB_D(2, epoch) = err_ratioB;
       
        % Test with test-data
       [test_errA, test_errB] = getError2(W_D, X_test, T_test);
       allTest_errA(i,epoch) = test_errA/size(A_test,2);
       allTest_errB(i,epoch) = test_errB/size(B_test,2);
    end
    
    sum_mean_train = sum(all_errAB_D,2)/nr_epochs;
    mean_train_errors(:,i) = sum_mean_train;
    
   
    
    %{
    % Check convergence
    current_errors = all_errAB_D(i,:);
    idx = find(current_errors < conv_criteria);
    if isempty(idx)
        disp('Delta algo. do not converge') 
    end
    if length(idx) < check_conv_points
        check_convergence_D = idx(1:length(idx));
    else 
        check_convergence_D = idx(1:check_conv_points);        
    end
    %}
    if rate == eta
        WD_final = W_D;
    end
   
    
end

mean_train_errorAB = sum(mean_train_errors,2)/max_iters
mean_test_errorsA = sum(allTest_errA,2)/nr_epochs;
mean_test_errorsB = sum(allTest_errB,2)/nr_epochs;
mean_test_errorAB = [sum(mean_test_errorsA,1); sum(mean_test_errorsB,1)]/max_iters



% Plot final decision boundary
% Final decision boundary Delta
x_2D = ((-WD_final(1)*x_1)-WD_final(3))/WD_final(2);
plot(x_1,x_2D,'g-')


% Plot comparisions between learning rates A, B
all_err1 = allTest_errA;
all_err2 = allTest_errB;
figure 
axis([0 nr_epochs -0.1 1])
for i = 1:max_iters
    hold on
    curve1 = all_err1(i,:);
    curve2 = all_err2(i,:);
    plot(1:length(curve1), curve1)
    plot(1:length(curve2), curve2)
end

legend('Error: Class A','Error: Class B','2','3')
xlabel('Epochs') % x-axis label
ylabel('Error rate ') % y-axis label
title('Error Rate for class A and B Independently')
%{
legend('Delta: \eta = 0.009','Delta: \eta = 0.003','Delta: \eta = 0.0001')
xlabel('Epochs') % x-axis label
ylabel('Error rate') % y-axis label
title('Covergence of the Delta algorithm for Not Linearly Separable Data')
%}





