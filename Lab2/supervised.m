clc
clear all 

%Train and test data 
X_train = [0 : 0.1 : 2*pi]';
%Y_train = sin(X_train);
Y_train = square(X_train);
X_dim = length(X_train); 

X_test = [0.05 : 0.1 : 2*pi]';
%Y_test = sin(X_test);
Y_test = square(X_test);
test_dim = length(X_test);

%plot functions 
%figure(1)
%plot (X_train, Y_train, 'r*')
%title('training data')
%hold on 
%plot (X_train, Y2_train, 'b')

e_train  = [];
e_test = [];
max_hidden = 30

for k = 1:max_hidden
    n_hidden = k; 
    means = linspace (0, 2*pi, n_hidden);
    %means = means(2 : end-1);

    %Hidden nodes matrix 
    Phi = zeros(X_dim , n_hidden);

    %TRAINING
    %Creation hidden nodes matrix 
    for i = 1:X_dim
        for j = 1:n_hidden
            Phi(i,j) = Gaussian (X_train(i), means(j));
        end
    end

    %Weights computation
    A = Phi'*Phi;
    b = Phi'*Y_train;
    w = A\b;

    %Output
    O = Phi*w;

    %Train error - averaged absolut difference 
    e = abs(O - Y_train);
    e_train = [e_train , (1/X_dim) * sum(e.^2)];

    %plot training output
%     figure(2)
%     plot (X_train, O)
%     title('train output')

    Phi = zeros(test_dim , n_hidden);
    %TESTING
    %Creation hidden nodes matrix 
    for i = 1:test_dim
        for j = 1:n_hidden
            Phi_t(i,j) = Gaussian (X_test(i), means(j));
        end
    end
    
    %output
    O_t = Phi_t*w;
 
%     figure(3)
%     plot(X_test, O_t)
%     title('test output')

    e_t = abs(O_t - Y_test);
    e_test = [e_test , (1/test_dim) * sum(e_t)];
end


figure (5)
plot([1:1:max_hidden], e_train, '-r')
hold on
plot([1:1:max_hidden], e_test, '-b')
legend ('train error', 'test error')
xlabel('number hidden nodes')
   








