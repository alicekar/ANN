clc 
clear all

%%

%Nodes_dim = 2;
n_epoch = 200;
%eta = 0.1;
alpha = 0.9;

%%
% Generate non separable data
n = 100;
dim_outputs = 1;
mA1 = [ -2.5 , 0.5]; mA2 = [2.5, 0.5]; sigmaA = 0.3;
mB = [ 0.0, -0.5]; sigmaB = 0.3;

% Data
classA(1, 1:n/2) = randn(1,n/2).*sigmaA + mA1(1);
classA(1, n/2+1:n) = randn(1,n/2).*sigmaA + mA2(1);
classA(2, 1:n/2) = randn(1,n/2).*sigmaA + mA1(2);
classA(2, n/2+1:n) = randn(1,n/2).*sigmaA + mA2(2);
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

%Bias
bias = ones(1,N);
X = [shuffled_classes; bias];
T = shuffled_targets;
X_dim = size(X,1);
T_dim = dim_outputs;

%%
% Plot classes  
figure
hold on;
axis([-3 3 -3 3]);
plot(classA(1,:), classA(2,:), 'b*');
plot(classB(1,:), classB(2,:), 'r*');
xlabel('x_1')
ylabel('x_2')
legend('classA', 'classB')

%%
n_epoch = 200;
misclassified = zeros(4, n_epoch);
error = zeros(4,5);
kk=0;
%Loop on the number of nodes in the hidden layer
Nodes_dim = 3;

for k = [0.01 0.1 0.2 0.4]
    
    kk = kk+1;
    %Nodes_dim = k;
    eta = k;
    % Initialize W
    mean = 0;
    std = 1/(N^(1/2));
    W = randn(Nodes_dim, X_dim).*std + mean;
    V = rands(T_dim, Nodes_dim +1).*std + mean; 

    dW = 0;
    dV = 0;

    %Loop on the epoch number
    for i = 1:n_epoch    

        %Forward Pass 
        hin = W * X;
        hout = [phi(hin); ones(1, N)];

        % Output Layer
        oin = V * hout;
        out = phi(oin);

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
        yfinal = sign(out);
    
        %Error vector
        e = T - yfinal;
        
        %Number of misclassified samples for different hidden nodes    
        misclassified(kk,i) = sum(abs(e./2));
        %MSE for different hidden nodes
        error(kk,i) = (1/N)*sum((T - out).^2);     
        
    end
       
end

%%
% misclassified plot
plot ([1:n_epoch], misclassified(1, :))
hold on
plot ([1:n_epoch], misclassified(2, :))
hold on
plot ([1:n_epoch], misclassified(3, :))
legend('2 nodes', '4 ndoes', '8 nodes')
title('# misclassified samples - eta=0.2 - alpha=0.9  ')

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

