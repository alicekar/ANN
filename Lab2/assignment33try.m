clc
clear all 

%Train data for sin(2x)

X_train = [0 : 0.1 : 2*pi]';
Y_train = sin(2*X_train);
X_dim = length(X_train); 

n_hidden = 2; 
CL_iter = 50; 

%Initialize weights 
W = 2*pi*abs(rands(1 , n_hidden)) ; %linspace(0,2*pi, n_hidden);  ;
Wnew = W;


figure(1)
plot (X_train , Y_train , '-')
hold on
% plot(W(:,1) , W(:,2), 'b*')
% hold on

hood_radius = 0;
%val = zeros(hood,1);
%idx = zeros(hood,1);
nu = 0.2;


for i = 1:CL_iter
    
    d = [];
    d_hood = [];
    
    %Choose a random sample from training set 
    index = randi (X_dim, 1, 1);
    train = [X_train(index) , Y_train(index)];
    
    %Distances of train sample from weights
    d = [ abs(train(1) - W).^2 ]; % , abs(train(2) - Gaussian(train(1) , W) ) ];
     
    %Find the winner // closer weights 

    winner = find(d == min(d)); 
    Winner = W(: , winner);
        
        
    idx = mod ( [winner-hood_radius : 1:winner+hood_radius]  ,  n_hidden );
    
    for j = 1:length(idx)
        if idx(j) == 0
            idx(j)=n_hidden;
        end
    end

    %Update weights 
    W( 1 , idx ) = W (1 , idx) + nu * (train(1)-W(1 , idx));
    %W = W / norm(W);

        
        
end    

%plot(Wnew(:,1) , Wnew(:,2), 'o')


%Presentation of Results
Y_final = zeros(X_dim, 1);
winner_final = zeros(X_dim , 1);
Winner_final = zeros(X_dim, 1);
for j = 1: n_hidden 
    plot (X_train, Gaussian (X_train, W(j)));
    hold on
end


for i = 1:X_dim
    
    d_final = [];
    train = [X_train(i) , Y_train(i)];

    %Distances of train sample from weights
    d_final = [ abs(train(1) - W).^2 ]; % , abs(train(2) - Gaussian(train(1) , W) ) ];
     
    %Find the winner // closer weights 

    winner_final = find(d_final == min(d_final)); 
    Winner_final = W(: , winner_final);
    
    
    Y_final(i) = Gaussian( train(1) , Winner_final );
    
end

%Final function approximation 
%plot (X_train , Y_final, '*')



