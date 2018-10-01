clc
clear all 

%Train data for sin(2x)

%X_train = [0 : 0.1 : 2*pi]';
%Y_train = sin(X_train);
%X_dim = length(X_train); 

n_hidden = 10; 
CL_iter = 10; 

%Initialize weights 

W(:,1) = abs(rands(n_hidden, 2));
hood = 0;
val = zeros(hood,1);
idx = zeros(hood,1);
nu = 0.2;


for i = 1:CL_iter
    
    d = [];
    d_hood = [];
    
    %Choose a random sample from training set 
    index = randi (X_dim, 1, 1);
    train = [X_train(index) , Y_train(index)];
    
    d1 = [ train(1) - W(:,1) , train(2) - W(:,2) ];
     
    %Computation of distances from weights and find the winner
    for j = 1:n_hidden
        
        d = [d ; d1(j, :)*d1(j,:)' ];
        winner = find(d == min(d)); 
        Winner = W(winner , :);
        
    end



   %Computation of distances from winner
    for j = 1:n_hidden 
        
        d_hood = [d_hood ; sum(abs(W(j, :) - Winner))];
        
    end 
    
    %Find neightbourhood index
    d_hood_change = d_hood;
    for j = 1:hood
        
         [val(j),idx(j)] = min(d_hood_change);
         % remove for the next iteration the last smallest value:
         d_hood_change(idx(j)) = 100; 
        
    end
    
    %Update weights 
    W( idx , 1 ) = W (idx , 1) + nu * (train(1)-W(idx , 1));
    W( idx , 2 ) = W (idx , 2) + nu * (train(2)-W(idx , 2));
    %plot(Wnew(:,1) , Wnew(:,2), 'o')
        
        
end    

%plot(Wnew(:,1) , Wnew(:,2), 'o')


%Presentation of Results
Y_final = zeros(X_dim, 1);
winner_final = zeros(X_dim , 1);
Winner_final = zeros(X_dim, 2);

for i = 1:X_dim
    
    d_final = [];
    train = [X_train(i) , Y_train(i)];
    d1_final = [ train(1) - W(:,1) , train(2) - W(:,2) ];
    
    for j = 1:n_hidden
        
        d_final = [d_final ; d1_final(j, :)*d1_final(j,:)' ];
        
    end
    
        winner_final(i) = find(d_final == min(d_final));
        Y_final(i) = Gaussian()
        Winner_final(i,:) = W(winner_final(i) , :);
    
end

%Final function approximation 









        