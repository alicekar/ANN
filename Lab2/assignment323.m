%% Data
X_train = train(: , 1:2);
Y_train = train(: , 3:4);
train_dim = size(X_train, 1);

X_test = test (: , 1:2);
Y_test = test (: , 3:4);
test_dim = size(X_test, 1);

%figure(1)
%plot(X_train(:,1), X_train(:,2), 'r*')
%hold on
%plot(X_test(:,1), X_test(:,2), 'b*')

%  figure 
%  hold on 
%  plot(Y_train(:,1), Y_train(:,2), 'r*')
%  plot(Y_test(:,1), Y_test(:,2), 'b*')

% Train

n_hidden = 10; 
n = n_hidden; 
CL_iter = 10;
Delta_iter = 10;
eta = 0.02;

%Initialize weights 
%w_min = min(X_train)
%w_max = max(X_train)
means_som = abs(rands(n_hidden , 2));
%plot (means_som(:,1) , means_som(:,2) , 'bo')
%hold on

hood = 2;
nu = 0.2;
sigmax = 20;
sigmay = sigmax;

for i = 1:CL_iter
    
    %Choose a random sample from training set 
    sample_index = randi(train_dim);
    sample = [X_train(sample_index,:) , Y_train(sample_index, :)];
    
    %Distances of train sample from weights
    d1 = sample(1) - means_som(:,1);
    d2 = sample(2) - means_som(:,2);
    d_som = d1.^2 + d2.^2 ; 
     
    %Find the winner // closer weights
     winner_som = find(d_som == min(d_som)); 
     Winner_som = means_som(winner_som , :);
     
     % Get indices of weights
    [sorted1, index1] = sort(means_som(:,1));
    [sorted2, index2] = sort(means_som(:,2));
    
    diff1 = abs(index1 - index1(winner_som));
    diff2 = abs(index2 - index2(winner_som));
    diff = diff1 + diff2;
    diff
     
    index_som = [];
    for i = 1:length(diff)
        if diff(i)<=hood
            index_som = [index_som i];
        end
    end
    
    index_som

    %Update weights 
    means_som( index_som , : ) = means_som (index_som , :) + nu * (sample(1:2)-means_som(index_som , :));
       
end    
 
 %plot(means_som(:,1), means_som(:,2), 'go')
 %legend ('inputs' , 'random initial means' , 'means after 100 SOM')



%2D DELTA SEQUENTIAL TRAINING AFTER SOM

% Shuffle
indices = randperm(train_dim);
X_train = X_train(indices, :);
Y_train = Y_train(indices, :);

e_test = [] ;
e_test_som = [];

    
    % TRAINING
    % Create the hidden nodes matrix Phi
    Phi = zeros(train_dim , n);
    for i = 1:train_dim
        for j = 1:n
            Phi(i,j) = twoGaussian(X_train(i,1), X_train(i,2), means_som(j,1), means_som(j,2), sigmax, sigmay);
        end
    end

    % TESTING
    % Create the hidden nodes matrix Phi
    Phi_t = zeros(test_dim , n);
    for i = 1:test_dim
        for j = 1:n
            Phi_t(i,j) = twoGaussian(X_test(i, 1), X_test(i,2), means_som(j, 1), means_som(j,2), sigmax, sigmay);
        end
    end 
   
    % Initialize weights
    w_start = normrnd(0, train_dim^(-1/2), [n,2]);
    w_start = w_start / norm(w_start)
    % Sequential training
    w = w_start;
  
 %epoch = 1;
    for epoch = 1 :Delta_iter
        
        for i=1:train_dim
        Phi_i = Phi(i,:);
        O_i = Phi_i*w;
        e_i_x = Y_train(i,1) - O_i(1);
        e_i_y = Y_train(i,2) - O_i(2);
        %e_i = Y_train(i, :) - O_i;
        delta_w_x = eta*e_i_x*Phi_i';
        delta_w_y = eta*e_i_y*Phi_i';
        %delta_w = eta * e_i * Phi_i';
        w(:,1) = w(:,1) + delta_w_x;
        w(:,2) = w(:,2) + delta_w_y;
        
        end
        w = w/norm(w);       
        % Shuffle 
        indices = randperm(train_dim);
        X_train = X_train(indices, :);
        Y_train = Y_train(indices, :);
        
        % Create the new hidden nodes matrix Phi
        Phi_t = zeros(train_dim, n);
        for i = 1:train_dim
            for j = 1:n
                Phi_t(i,j) = twoGaussian(X_test(i, 1), X_test(i,2), means_som(j, 1), means_som(j,2), sigmax, sigmay);
            end
        end
        
    % Output
    O_t_som = Phi_t*w;
    
    % Test error - absolute difference 
    e_t_som = abs(O_t_som - Y_test);
    e_sum = mean(mean(e_t_som));
    e_test_som = [e_test_som  e_sum];
    % e_test_som(epoch) = mean(mean(abs(O_t_som - Y_test)))   ;

    end

    % Output
    %O_final_som = O_t_som;
plot ([1:1:Delta_iter] , e_test_som)
















