%% Import data
% Author: Elin

clc
clear all 

data = importdata('pict.dat');
data = reshape(data,1024,11);
data = data';
x = data(1:3,:)';
N = 1024;
P = 3;

% Plot the input data
figure;
hold on;
for i = 1:P
    subplot(1,3,i);
    result = mat2gray(reshape(x(:,i),32,32));
    imshow(result,'InitialMagnification', 1000);
    title(['Picture ' num2str(i)])
end
suptitle('Input patterns')

%% Train with the little model

% Train
W = zeros(N, N);
for i = 1:P
    W = W + (1/N)*x(:,i)*x(:,i)';
end

%% Test with noisy data 

noise = 1/4;

figure;
hold on;

for pat = 1:P
    x_test = x(:,pat);
    x_goal = x_test;

    flip_indices = randperm(N,N*noise);
    for i= 1:N*noise
        x_test(flip_indices(i)) = (-1)*x_test(flip_indices(i));
    end

    % Test
    change = 1;
    count = 0;
    while change>0
        x_old = x_test;
        x_test = sign(W * x_old);
        change = sum(sum(abs(x_test-x_old)));
        count = count + 1;
    end

    error = sum(abs(x_test-x_goal))/2;

    % Plot the test result
    subplot(1,3,pat);
    result = mat2gray(reshape(x_test,32,32));
    imshow(result,'InitialMagnification', 1000);
    title(['Picture ' num2str(pat)])
end
suptitle('Output patterns')