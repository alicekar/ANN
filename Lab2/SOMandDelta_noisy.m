%%SOM ALGORITHM

%clc
%clear all 

%Train data for sin(2x)
X_train = [0 : 0.1 : 2*pi]';
train_dim = length(X_train);
Y_train = sin(2*X_train)+ normrnd(0, 0.1, [train_dim,1]);
 

%Test data 
% Create test data
X_test = [0.05 : 0.1 : 2*pi]';
test_dim = length(X_test);
Y_test = sin(2*X_test)+ normrnd(0, 0.1, [test_dim,1]);


n_hidden = 10; 
n = n_hidden; 
CL_iter = 15;
Delta_iter = 100;
eta = 0.02;

%Initialize weights 
%means_som = 2*pi*abs(rands(1 , n_hidden));%linspace(0,2*pi, n_hidden); %2*pi*abs(rands(1 , n_hidden)) ;
%means = means_som;

%figure(1)
%plot (X_train , Y_train , '-')
%hold on
% plot(W(:,1) , W(:,2), 'b*')
% hold on

hood_radius = 1;
nu = 0.2;
sigma = 1.3;


for i = 1:CL_iter
    
    d_som = [];
        
    %Choose a random sample from training set 
    index = randi (train_dim, 1, 1);
    train = [X_train(index) , Y_train(index)];
    
    %Distances of train sample from weights
    d_som = abs(train(1) - means_som).^2;
     
    %Find the winner // closer weights
    winner_som = find(d_som == min(d_som)); 
    Winner_som = means_som(: , winner_som);
        
    % if (winner_som+hood_radius > n_hidden)
    %     upper = n_hidden;
    % else 
         upper = winner_som +hood_radius;
    % end
     
    % if (winner_som-hood_radius < 1)
    %     down = 1;
    % else
         down = winner_som-hood_radius;
    % end
    
    idx_som = mod ( [down : 1 : upper ]  ,  n_hidden );
    
    for j = 1:length(idx_som)
        if idx_som(j) == 0
            idx_som(j)=n_hidden;
        end
    end

    %Update weights 
    means_som( 1 , idx_som ) = means_som (1 , idx_som) + nu * (train(1)-means_som(1 , idx_som));
        
        
end    


%Plot of final radius functions
% for j = 1: n_hidden 
%     plot (X_train, Gaussian (X_train, M(j)));
%     hold on
% end

%DELTA SEQUENTIAL TRAINING AFTER SOM

% Shuffle
indices = randperm(train_dim);
X_train = X_train(indices);
Y_train = Y_train(indices);

e_test = [] ;
e_test_som = [];


     
    % TRAINING
    % Create the hidden nodes matrix Phi
    Phi = zeros(train_dim , n);
    for i = 1:train_dim
        for j = 1:n
            Phi(i,j) = Gaussian(X_train(i), means_som(j), sigma);
        end
    end
    
    % TESTING
    % Create the hidden nodes matrix Phi
    Phi_t = zeros(test_dim , n);
    for i = 1:test_dim
        for j = 1:n
            Phi_t(i,j) = Gaussian(X_test(i), means_som(j), sigma);
        end
    end 
    
    % Initialize weights
    w_start = normrnd(0, train_dim^(-1/2), [n,1]);
    % Sequential training
    w = w_start;
    
    for epoch = 1:Delta_iter
        
        for i=1:train_dim
            Phi_i = Phi(i,:);
            O_i = Phi_i*w;
            e_i = Y_train(i) - O_i;
            delta_w = eta*e_i*Phi_i';
            w = w + delta_w;
        end
               
        % Shuffle 
        indices = randperm(train_dim);
        X_train = X_train(indices);
        Y_train = Y_train(indices);
        
        % Create the new hidden nodes matrix Phi
        Phi = zeros(train_dim, n);
        for i = 1:train_dim
            for j = 1:n
                Phi(i,j) = Gaussian(X_train(i), means_som(j), sigma);
            end
        end
        
    % Output
    O_t_som = Phi_t*w;
    
    % Test error - average absolute difference 
    e_t_som = abs(O_t_som - Y_test);
    e_test_som = [e_test_som  mean(e_t_som)];
        
    end

    % Output
    O_final_som = O_t_som;
    
    % Test error - average absolute difference 
    %e_t = abs(O_t - Y_test);
    %e_test = [e_test  mean(e_t)];
%     e_t_clean = abs(O_t - Y_test);
%     e_test_clean = [e_test_clean  mean(e_t_clean)];



%plot (X_test, O_final_som , '*')
    
%figure(2)
plot ([1:1:Delta_iter], e_test_som, 'r--')
hold on
    
   
% DELTA SEQUENTIAL TRAINING WITHOUT SOM  

% Shuffle
indices = randperm(train_dim);
X_train = X_train(indices);
Y_train = Y_train(indices);

e_test = [] ;

%Initialize weights 
%means = linspace(0,2*pi, n_hidden);

     
    % TRAINING
    % Create the hidden nodes matrix Phi
    Phi = zeros(train_dim , n);
    for i = 1:train_dim
        for j = 1:n
            Phi(i,j) = Gaussian(X_train(i), means(j), sigma);
        end
    end
    
    % TESTING
    % Create the hidden nodes matrix Phi
    Phi_t = zeros(test_dim , n);
    for i = 1:test_dim
        for j = 1:n
            Phi_t(i,j) = Gaussian(X_test(i), means(j), sigma);
        end
    end 
    
    % Initialize weights
    w_start = normrnd(0, train_dim^(-1/2), [n,1]);
    % Sequential training
    w = w_start;
    
    for epoch = 1:Delta_iter
        
        for i=1:train_dim
            Phi_i = Phi(i,:);
            O_i = Phi_i*w;
            e_i = Y_train(i) - O_i;
            delta_w = eta*e_i*Phi_i';
            w = w + delta_w;
        end
               
        % Shuffle 
        indices = randperm(train_dim);
        X_train = X_train(indices);
        Y_train = Y_train(indices);
        
        % Create the new hidden nodes matrix Phi
        Phi = zeros(train_dim, n);
        for i = 1:train_dim
            for j = 1:n
                Phi(i,j) = Gaussian(X_train(i), means(j), sigma);
            end
        end
        
    % Output
    O_t = Phi_t*w;
    
    % Test error - average absolute difference 
    e_t = abs(O_t - Y_test);
    e_test = [e_test  mean(e_t)];
        
    end

    % Output
    O_final = O_t;
    
    % Test error - average absolute difference 
    %e_t = abs(O_t - Y_test);
    %e_test = [e_test  mean(e_t)];
%     e_t_clean = abs(O_t - Y_test);
%     e_test_clean = [e_test_clean  mean(e_t_clean)];



    %plot (X_test, O_t , '*')
    

    plot ([1:1:Delta_iter], e_test, 'b--')
    legend ('SOM&Delta' , 'Delta' , 'noisy SOM&Delta' , 'noisy Delta')
    xlabel('Delta epoch iteration')
    ylabel('abs(mena(error))')
    

