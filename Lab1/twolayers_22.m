clc 
clear all

%
% Generate different version of linearly non-separable data
dim_outputs = 1;
ndata = 100;
n = ndata;
mA1 = [ -2.5 , 0.5]; mA2 = [2.5, 0.5]; sigmaA = 0.3;
mB = [ 0.0, -0.5]; sigmaB = 0.3;

classA(1, 1:n/2) = randn(1,n/2).*sigmaA + mA1(1);
classA(1, n/2+1:n) = randn(1,n/2).*sigmaA + mA2(1);
classA(2, 1:n/2) = randn(1,n/2).*sigmaA + mA1(2);
classA(2, n/2+1:n) = randn(1,n/2).*sigmaA + mA2(2);
classB(1,:) = randn(1,n).*sigmaB + mB(1);
classB(2,:) = randn(1,n).*sigmaB + mB(2);
targetsA = ones(1,n);
targetsB = ones(1,n)*-1;

dataA = [classA; targetsA];
dataB = [classB; targetsB];

% Shuffle both classes separately 
dataA_shuffled = dataA(:,randperm(ndata));
dataB_shuffled = dataB(:,randperm(ndata));

% DIFFERENT SUBSAMPLES 
% Choose amount of data to remove from each class (i.e amount of test data)
percentageA = 00; 
percentageB = 50;
n_train = 2 * ndata - percentageA - percentageB;
n_test = 2*ndata - n_train;
A_train = dataA_shuffled(:,1:(ndata-percentageA));
A_test = dataA_shuffled(:,(ndata-percentageA+1):ndata);
B_train = dataB_shuffled(:,1:(ndata-percentageB));
B_test = dataB_shuffled(:,(ndata-percentageB+1):ndata);
data_train = [A_train, B_train];
data_test = [A_test, B_test];
data_train_shuffled = data_train(:,randperm(size(data_train,2)));
data_test_shuffled = data_test(:,randperm(size(data_test,2)));
X_train = [data_train_shuffled(1:2,:); ones(1,size(data_train,2))]; % +bias
T_train = data_train_shuffled(3,:);
X_test = [data_test_shuffled(1:2,:); ones(1,size(data_test,2))];    % +bias
T_test = data_test_shuffled(3,:);
A_25_50 = A_train(1:2,:);
B_25_50 = B_train(1:2,:);

% Choose which data to use to train
X = X_train;
T = T_train;
X_dim = size(X,1);
T_dim = dim_outputs;
%pointsA = A_20_80; %A_20_80;
%pointsB = B_20_80; %B_20_80;
%X_test = X_20_80_test;
%T_test = T_20_80_test;
nn = size(X);
n_train = nn(2);
nn = size(X_test);
n_test = nn(2);

% %% PART 2  - 20-80 case
% clc 
% clear all 
% close all
%  
% % 
% dim_outputs = 1;
% ndata = 100;
% n = ndata;
% % mA = [ 1.0, 0.3];    sigmaA = 0.2;
% % mB = [ 0.0, -0.1];   sigmaB = 0.3;
% mA1 = [ -2.5 , 0.5]; mA2 = [2.5, 0.5]; sigmaA = 0.3;
% mB = [ 0.0, -0.5]; sigmaB = 0.3;
% %classA(1,:) = [ randn(1,round(0.5*ndata)) .* sigmaA - mA(1)   ,    randn(1,round(0.5*ndata)) .* sigmaA + mA(1)];
% %classA(2,:) = randn(1,ndata) .* sigmaA + mA(2);
% classA(1, 1:n/2) = randn(1,n/2).*sigmaA + mA1(1);
% classA(1, n/2+1:n) = randn(1,n/2).*sigmaA + mA2(1);
% classA(2, 1:n/2) = randn(1,n/2).*sigmaA + mA1(2);
% classA(2, n/2+1:n) = randn(1,n/2).*sigmaA + mA2(2);
% classB(1,:) = randn(1,ndata) .* sigmaB + mB(1);
% classB(2,:) = randn(1,ndata) .* sigmaB + mB(2);
% targetsA = ones(1,ndata);
% targetsB = ones(1,ndata)*-1;
% dataA = [classA; targetsA];
% dataB = [classB; targetsB];
%  
% % Shuffle both classes separately 
% dataA_shuffled = dataA(:,randperm(ndata));
% dataB_shuffled = dataB(:,randperm(ndata));
% % Keep 80% from a subset of classA for which classA(1,:)<0 and 20% from a 
% % subset of classA for which classA(1,:)>0
% % 50 first of classA < 0, 50 last > 0
% A_lessThan0 = dataA(:,1:size(dataA,2)/2);
% idx_less = 1:size(A_lessThan0,2);
% mix1_idx = idx_less(:,randperm(length(idx_less)));
% stop_lessThan0 = 0.8*length(mix1_idx);
% idx1_train = mix1_idx(:,1:stop_lessThan0);
% idx1_test = mix1_idx(:,stop_lessThan0+1:end);
% A_train_lessThan0 = A_lessThan0(:,idx1_train);
% A_test_lessThan0 = A_lessThan0(:,idx1_test);
%  
% A_greaterThan0 = dataA(:,size(dataA,2)/2+1:end);
% idx_greater = 1:size(A_greaterThan0,2);
% mix2_idx = idx_greater(:,randperm(length(idx_greater)));
% stop_greaterThan0 = 0.2*length(mix1_idx);
% idx2_train = mix2_idx(:,1:stop_greaterThan0);
% idx2_test = mix2_idx(:,stop_greaterThan0+1:end);
% A_train_greaterThan0 = A_greaterThan0(:,idx2_train);
% A_test_greaterThan0 = A_greaterThan0(:,idx2_test);
%  
% dataA_20_80_train = [A_train_lessThan0 A_train_greaterThan0];
% dataA_20_80_test = [A_test_lessThan0 A_test_greaterThan0];
% A_20_80 = dataA_20_80_train(1:2,:);
% B_20_80 = classB;
% data_train_20_80 = [dataA_20_80_train dataB];
% data_train_20_80_shuffled = data_train_20_80(:,randperm(size(data_train_20_80,2)));
% X_20_80_train = [data_train_20_80_shuffled(1:2,:); ones(1,size(data_train_20_80,2))];
% T_20_80_train = [data_train_20_80_shuffled(3,:)];
% data_test_20_80_shuffled = dataA_20_80_test(:,randperm(size(dataA_20_80_test,2)));
% X_20_80_test = [data_test_20_80_shuffled(1:2,:); ones(1,size(dataA_20_80_test,2))];
% T_20_80_test = [data_test_20_80_shuffled(3,:)];
% 
% 
% % Choose which data to use to train
% X = X_20_80_train;
% T = T_20_80_train;
% X_dim = size(X,1);
% T_dim = dim_outputs;
% pointsA = A_20_80; %A_20_80;
% pointsB = B_20_80; %B_20_80;
% X_test = X_20_80_test;
% T_test = T_20_80_test;
% nn = size(X);
% n_train = nn(2);
% nn = size(X_test);
% n_test = nn(2);
%
% % Plot classes  
% figure
% hold on;
% axis([-3 3 -3 3]);
% plot(classA(1,:), classA(2,:), 'b*');
% plot(classB(1,:), classB(2,:), 'r*');
% xlabel('x_1')
% ylabel('x_2')
% legend('classA', 'classB')



n_epoch = 50;
misclassified_train = zeros(3, n_epoch);
error_train = zeros(4,5);
misclassified_test = zeros(3, n_epoch);
error_test = zeros(4,5);
kk=0;
%Loop on the number of nodes in the hidden layer
%Nodes_dim = 3;
eta = 0.2;
alpha = 0.9;
hidden = [4];

for k = hidden
    
    kk = kk+1;
    Nodes_dim = k;
    %eta = k;
    
    % Initialize W
    mean = 0;
    std = 1/(n_train^(1/2));
    W = randn(Nodes_dim, X_dim).*std + mean;
    V = rands(T_dim, Nodes_dim +1).*std + mean; 

    dW = 0;
    dV = 0;

    %Loop on the epoch number
    for i = 1:n_epoch    

        %Forward Pass for training samples
        hin = W * X;
        hout = [phi(hin); ones(1, n_train)];
        
        %Forward Pass for test samples
        hin_t = W * X_test;
        hout_t = [phi(hin_t) ; ones(1, n_test)];

        % Output Layer for training samples
        oin = V * hout;
        out = phi(oin);
        
        % Output Layer for test samples
        oin_t = V * hout_t;
        out_t = phi(oin_t);

        % 2. Backward pass
        delta_o = (out - T) .* phiprime(out);
        delta_h = (V' * delta_o) .* phiprime(hout);
        delta_h = delta_h(1:Nodes_dim, :); % remove bias term

        % 3. Weight update
        dW = (dW .* alpha) - (delta_h * X') .* (1 - alpha);
        dV = (dV .* alpha) - (delta_o * hout') .* (1 - alpha);
        W = W + dW .* eta;
        V = V + dV .* eta;
        
        %Computation of y_final
        yfinal_train = sign(out);
        yfinal_test = sign(out_t);
    
        %Error vector
        e_train = T - yfinal_train;
        e_test = T_test - yfinal_test;
        
        %Number of misclassified samples for different hidden nodes    
        misclassified_train(kk,i) = (1/n_train)*sum(abs(e_train./2));
        misclassified_test(kk,i) = (1/n_test)*sum(abs(e_test./2));
        
        %MSE for different hidden nodes
        error_train(kk,i) = (1/n_train)*sum((T - out).^2);   
        error_test(kk,i) = (1/n_test)*sum((T_test - out_t).^2);
        
    end
       
end


%
% 
% % Plot decision boundary  
% 
% XXX1 = linspace(-3 , +3 , 10 );
% XXX2 = linspace( -3 , +3 , 10);
% XX2 = [];
% XX1 = [];
% %XX2 = zeros(1,10);
% 
% for i=1:10
%      XX1= [XX1 , XXX1];
%      XX2 = [XX2 , XXX2(i)*ones(1,10)];
% end
% 
% XX = [XX1 ; XX2 ; ones(1, 100)];
% 
% np = length(XX1);
% hin_p = W * XX;
% hout_p = [phi(hin_p); ones(1, np)];
% oin_p = V * hout_p;
% out_p = phi(oin_p);
% 
% % matrix form of out_p
% 
% OUT = zeros (10, 10);
% for i = 1:10
%     for j = 1:10
%         OUT(i,j) = out_p( (i-1)*10 + j );
%     end
% end
% 
% contour(XXX1, XXX2, OUT)
% xlabel('x_1')
% ylabel('x_2')
% %legend('classA', 'classB')






%% misclassified plot
figure
%subplot(1, 3, 1)
plot ([1:n_epoch], misclassified_train(1, :))
hold on
plot ([1:n_epoch], misclassified_test(1, :))
xlabel('n epochs')
ylabel('miscassified samples')
legend('train', 'test')
title('hidden = 4 - 00-50 case')
%%
subplot(1, 3, 2)
plot ([1:n_epoch], misclassified_train(2, :))
hold on
plot ([1:n_epoch], misclassified_test(2, :))
legend('train', 'test')
title('hidden = 8 ')

subplot(1, 3, 3)
plot ([1:n_epoch], misclassified_train(3, :))
hold on
plot ([1:n_epoch], misclassified_test(3, :))
legend('train', 'test')
title('hidden = 16 ')


%%
% misclassified plot on eta
plot ([1:n_epoch], misclassified(1, :))
hold on
plot ([1:n_epoch], misclassified(2, :))
hold on
plot ([1:n_epoch], misclassified(3, :))
hold on
plot ([1:n_epoch], misclassified(4, :))
legend('eta=0.1', 'eta=0.2', 'eta=0.4', 'eta=0.8')
title('#misclassified samples -#nodes=4 - epochs=200 - alpha=0.9')

%%

% MSE plot
plot ([1:n_epoch], error(1, :))
hold on
plot ([1:n_epoch], error(2, :))
hold on
plot ([1:n_epoch], error(3, :))
legend('2 nodes', '4 ndoes', '8 nodes')
title('MSE - eta=0.2 - alpha=0.9  ')
