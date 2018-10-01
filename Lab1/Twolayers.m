clc
clear all

%Data Creation 
[x,t] = not_sep_data(100);

n_maxnodes = [5, 10, 20, 30, 50];
hidden_nodes = [1:n_maxnodes];
e = zeros(1, 5);

yfinal = zeros(1,200);
z=0;

for i = n_maxnodes
    z = z+1;
    net = newff(minmax(x), [3 3 1], {'purelin','purelin', 'purelin'}, 'traingdm' );
    %net = newff(minmax(x), [i 1], {'purelin','purelin'}, 'trainlm' );    
    net = train(net,x,t);
    
    y = sim(net, x);    
    
    for j = 1:length(y)
        if y(j) > 0
            yfinal(j) = 1
        elseif y(j) <= 0
            yfinal(j) = 0
        end
    end
    
    e(z) = sum(abs(t-yfinal));
end

plot (hidden_nodes, e)

%% PART 2  

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
percentageA = 50; 
percentageB = 0;
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


% Keep 80% from a subset of classA for which classA(1,:)<0 and 20% from a 
% subset of classA for which classA(1,:)>0
Aindex_lessThan0 = find(dataA_shuffled(1,:)<0);      % 50 points
Aindex_greaterThan0 = find(dataA_shuffled(1,:)>0);
A80_lessThan0 = dataA_shuffled(:,Aindex_lessThan0(:,randperm(40)));   % keep 80% < 0
A20_greaterThan0 = dataA_shuffled(:, Aindex_greaterThan0(:,randperm(10))); % keep 20% > 0

dataA_20_80 = [A80_lessThan0, A20_greaterThan0];
data_20_80 = [A80_lessThan0, A20_greaterThan0, dataB];
data_20_80_shuffled = data_20_80(:, randperm(size(data_20_80,2)));
X_20_80 = [data_20_80_shuffled(1:2,:); ones(1,size(data_20_80,2))];  % +bias
T_20_80 = data_20_80_shuffled(3,:);


% Choose which data to use
X = X_20_80;
T = T_20_80;

% Plot, choose which data
%pointsA = A_train;
%pointsB = B_train;
%figure
%hold on;
%axis([-2 2 -2 2]);
%plot(pointsA(1,:), pointsA(2,:), 'b*');
%plot(pointsB(1,:), pointsB(2,:), 'r*');