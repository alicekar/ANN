% Author: Alice

% 3.3 Function Approximaion
%% 3.3.1 Generate function data
% Gauss function
clc 
clear all 
close all

x=[-5:0.5:5]';
y=[-5:0.5:5]';
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
%{
figure;
mesh(x, y, z);
axis([-5 5 -5 5 -0.7 0.7]);
xlabel('x')
ylabel('y')
zlabel('z')
%}
ndata = length(x)*length(y);
targets = reshape(z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns =[reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

% Add Bias
X = [patterns; ones(1,size(patterns,2))];
T = targets; 
N = size(X,2);
data = [X; T];

% Decide amount of training data
%324/N %400/N  %  ?0.52, ?0.73, ?0.9 %For plot
sets = [0.8 0.7 0.6 0.5 0.4 0.3 0.2];

% Parameters
eta = 0.009;
nr_epochs = 100;
nr_iterations = 10;
alpha = 0.9;
nodes = 22;%1:1:25;

% Train the network and visualise the approximated function
mse_test_different_sets = zeros(1, length(sets));
for s = 1:length(sets)
    set = sets(s);
    amount_train = set;
    
    % Divide into train and test data
    stop = round(amount_train*size(X,2));
    indices = randperm(N);
    shuffled_data = data(:,indices);
    X_train = shuffled_data(1:size(data,1)-1, 1:stop);
    T_train = shuffled_data(size(data,1), 1:stop);
    %X_test = shuffled_data(1:size(data,1)-1, stop+1:end);
    %T_test = shuffled_data(size(data,1), stop+1:end);
    X_test = shuffled_data(1:size(data,1)-1,:);
    T_test = shuffled_data(size(data,1),:);

    %{
    % Only for plotting (Select relevant part training)
    x_train = shuffled_data(1,1:stop);
    y_train = shuffled_data(2,1:stop);
    gridsize = sqrt(length(x_train));
    xx = reshape(x_train, gridsize, gridsize);
    yy = reshape(y_train, gridsize, gridsize);
    %}
    %{
    % For plotting classic data with mesh
    X_train = X;
    T_train = T;
    xx = x;
    yy = y;
    gridsize = length(xx);
    %}

    % 2-layer perceptron
    n_train = size(X_train,2);
    n_test = size(X_test,2);
    iter_errors_train = zeros(length(nodes),nr_epochs,nr_iterations);
    iter_errors_test = zeros(nr_iterations,length(nodes));
    for iter = 1:nr_iterations
        errors_train = zeros(length(nodes),nr_epochs);
        errors_test = zeros(1,length(nodes));
        for i = 1:length(nodes)

            % Initialize W
            Nhidden = nodes(i);
            T_dim = size(T_train,1);
            X_dim = size(X_train,1);
            dW = zeros(Nhidden, X_dim);
            dV = zeros(T_dim, Nhidden+1);
            W = normrnd(0,1,[Nhidden X_dim]);
            V = normrnd(0,1,[T_dim Nhidden+1]); % Bias

            %figure
            for epoch = 1:nr_epochs
                % Forward pass
                hin = W * X_train;
                hout = [2 ./ (1+exp(-hin)) - 1 ; ones(1,n_train)];
                oin = V * hout;
                out = 2 ./ (1+exp(-oin)) - 1;

                % Backward pass  o=y
                delta_o = (out - T_train) .* ((1 + out) .* (1 - out)) * 0.5;
                delta_h = (V' * delta_o) .* ((1 + hout) .* (1 - hout)) * 0.5;
                delta_h = delta_h(1:Nhidden, :);

                % Weight update
                dW = (dW .* alpha) - (delta_h * X_train') .* (1-alpha);
                dV = (dV .* alpha) - (delta_o * hout') .* (1-alpha);
                W = W + dW .* eta;
                V = V + dV .* eta;

                % Animated plot
                %zz = reshape(out, gridsize, gridsize);
                %mesh(xx,yy,zz);
                %axis([-5 5 -5 5 -0.7 0.7]);
                %plot(xx,zz,'r*')
                %drawnow;

                % MSE Error train
                err_train = sum((T_train-out).^2)*(1/n_train);
                errors_train(i,epoch) = err_train;

            end
            % Test
            hin_test = W * X_test;
            hout_test = [2 ./ (1+exp(-hin_test)) - 1 ; ones(1,n_test)];
            oin_test = V * hout_test;
            out_test = 2 ./ (1+exp(-oin_test)) - 1;

            % MSE Error
            err_test = sum((T_test-out_test).^2)*(1/n_test);
            errors_test(i) = err_test;
        end
        iter_errors_train(:,:,iter) = errors_train;
        iter_errors_test(iter,:) = errors_test;
    end
    % Find number of nodes that minimizes the error
    mean_errors_test = sum(iter_errors_test,1)/nr_iterations
    mean_errors_train = sum(iter_errors_train,3)/nr_iterations;
    final_errors_train = mean_errors_train(:,end);
    min_train = min(final_errors_train);
    best_nrNodes_train = find(final_errors_train == min_train)
    min_train = min(mean_errors_test);
    best_nrNodes_test = find(mean_errors_test == min_train)
    
    mse_test_different_sets(s) = mean_errors_test
end


figure
plot(sets, mse_test_different_sets)
title('MSE Error for Test-set when Varying the Training-set')
xlabel('Amount of Training data')
ylabel('MSE for Test data')

%{
figure
axis([0 nr_epochs 0 1])
errors = mean_errors_train;
for j = 5:5:length(nodes)
    hold on
    plot(1:nr_epochs, errors(j,:))
end
title('Covergence of the 2-layer Perceptron Algorithm for Function Approx.')
legend('5 Nodes', '10 Nodes', '15 Nodes', '20 Nodes', '25 Nodes')
xlabel('Epochs') % x-axis label
ylabel('MSE Error') % y-axis label   
%}


