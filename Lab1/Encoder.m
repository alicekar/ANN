%Data generation - ENCODER PROBLEM 
clc
clear all

patterns = -ones(8,8);
dd = 2*ones(1,8);
d = diag(dd);
patterns = patterns + d;
bias = ones(1,8);
X = [patterns ; bias];
T = patterns;
n_train = 8;
dim_outputs = 8;
X_dim = size(X,1);
T_dim = dim_outputs;

%%
n_epoch = 1000;
misclassified_train = zeros(3, n_epoch);
error_train = zeros(3,n_epoch);
misclassified_test = zeros(3, n_epoch);
error_test = zeros(3,n_epoch);
kk=0;

%Loop on the number of nodes in the hidden layer
%Nodes_dim = 3;
eta = 0.2;
alpha = 0.9;
hidden = [3];

for k = [0.1]
    
    kk = kk+1;
    Nodes_dim = 3;
    eta = k;
    
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
        %hin_t = W * X_test;
        %hout_t = [phi(hin_t) ; ones(1, n_test)];

        % Output Layer for training samples
        oin = V * hout;
        out = phi(oin);
        
        % Output Layer for test samples
        %oin_t = V * hout_t;
        %out_t = phi(oin_t);

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
        %yfinal_test = sign(out_t);
    
        %Error vector
        e_train = T - yfinal_train;
        %e_test = T_test - yfinal_test;
        
        %Number of misclassified samples for different hidden nodes    
        misclassified_train(kk,i) = sum(sum(abs(e_train./2)));%(1/n_train)*sum(sum(abs(e_train./2)));
        %misclassified_test(kk,i) = (1/n_test)*sum(abs(e_test./2));
        
        %MSE for different hidden nodes
        error_train(kk,i) = (1/n_train)*sum(sum((T - out).^2));   
        %error_test(kk,i) = (1/n_test)*sum((T_test - out_t).^2);
        
        if (misclassified_train(kk,i) == 0)
            break; 
        end
        
    end
       
end

%%

% misclassified plot
figure
%subplot(1, 3, 1)
plot ([1:n_epoch], misclassified_train(1, :))
hold on
plot ([1:n_epoch], misclassified_train(2, :))
hold on 
plot ([1:n_epoch], misclassified_train(3, :))
legend('eta = 0.01', 'eta = 0.05' , 'eta = 0.1')
title('Encoder 2 ')



% subplot(1, 3, 2)
% plot ([1:n_epoch], misclassified_train(2, :))
% hold on
% plot ([1:n_epoch], misclassified_test(2, :))
% legend('train', 'test')
% title('hidden = 8 ')
% 
% subplot(1, 3, 3)
% plot ([1:n_epoch], misclassified_train(3, :))
% hold on
% plot ([1:n_epoch], misclassified_test(3, :))
% legend('train', 'test')
% title('hidden = 16 ')