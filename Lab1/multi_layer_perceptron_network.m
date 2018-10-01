% _______________________________4.3.0______________________________________ 
%% Generate data 

format shortE

% Generate Mackay-Glass data points with Eulers method
data = zeros(1, 1700);                     % -100 to 1600
data(100) = 1.5;                           % x_0
for i=100:1699
    data(i+1)= data(i) + 0.2*data(i-25)/(1 + (data(i-25))^10) - 0.1*data(i);
end

% Generate input and output data (x and t)
n = 1200;
x = zeros(5,n);
t = zeros(1,n);
for i=301:1500
    x(:,i-300) = [data(i-20), data(i-15), data(i-10), data(i-5), data(i)];
    t(i-300) = data(i+5);
end

figure
plot(linspace(300,1500,1200), t)
title({['Mackey-Glass Time Series']})
ylabel('Time series x(t)')
xlabel('Time step t')

figure
t_noise = t + normrnd(0, 0.18, [1,n]);
plot(linspace(300,1500,1200), t_noise)
title({['Mackey-Glass Time Series with Maximum Noise']})
ylabel('Time series x(t)')
xlabel('Time step t')

% _______________________________4.3.1______________________________________ 
%% Two-layer perceptron for time series predicition 

% Choose a Training Function (help nntrain)
trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation.

% Early stopping
%net.trainParam.max_fail = 6;    % Maximum validation failures

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';          % No shuffling, we want consecutive blocks
net.divideMode = 'sample';
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;

% Choose a Performance Function (help nnperformance)
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions (help nnplot)
net.plotFcns = {'plotperform', 'plotfit'};

% Configurations
reg_strength = [0.0, 0.25, 0.5, 0.75 , 1.0];
hidden = [2, 4, 6, 8];

% Calculate average mse and std for each configuration 
av_train_mse_matrix = zeros(length(reg_strength), length(hidden));
av_val_mse_matrix = zeros(length(reg_strength), length(hidden));
av_test_mse_matrix = zeros(length(reg_strength), length(hidden));

std_train_mse_matrix = zeros(length(reg_strength), length(hidden));
std_val_mse_matrix = zeros(length(reg_strength), length(hidden));
std_test_mse_matrix = zeros(length(reg_strength), length(hidden));

best_test = 1000;

for r = 1:length(reg_strength)
    weights = [];
    for h = 1:length(hidden) 
        % Set strength of regularization 
        net.performParam.regularization = reg_strength(r);
        % Set number of hidden nodes
        hiddenLayerSize = hidden(h);

        % Create a Fitting Network 
        net = fitnet(hiddenLayerSize,trainFcn);
        
        % Initialize mse for this configuration
        train_mse = [];
        val_mse = [];
        test_mse = [];

        % Repeat for valid statistics
        repeats = 100;
        for i = 1:repeats;
            % Initalize
            net = init(net);

            % Train the Network
            [net,tr] = train(net,x,t);

            % Test the Network
            y = net(x);
            e = gsubtract(t,y);
            %performance = perform(net,t,y);
            nntraintool close;

            % Recalculate Training, Validation and Test Performance
            trainTargets = t .* tr.trainMask{1};
            trainPerformance = perform(net,trainTargets,y);
            train_mse = [train_mse trainPerformance];

            valTargets = t .* tr.valMask{1};
            valPerformance = perform(net,valTargets,y);
            val_mse = [val_mse valPerformance];

            testTargets = t .* tr.testMask{1};
            testPerformance = perform(net,testTargets,y);
            test_mse = [test_mse testPerformance];
            
            weights = [weights; net.iw{1,1}];
            
        end
        av_train_mse_matrix(r,h) = mean(train_mse);
        av_val_mse_matrix(r,h) = mean(val_mse);
        av_test_mse_matrix(r,h) = mean(test_mse);  
        
        std_train_mse_matrix(r,h) = std(train_mse);
        std_val_mse_matrix(r,h) = std(val_mse);
        std_test_mse_matrix(r,h) = std(test_mse);  
        
        % Plot a test prediction for an example model of better configuration
        if av_val_mse_matrix(r,h) < best_test
            figure
            hold on
            plot(linspace(1261, 1500, 240), y(961:1200));
            plot(linspace(1261, 1500, 240), t(961:1200));
            title({['Best Model']; ['Strength of Regularisation: ' num2str(reg_strength(r))]; ['Number of Hidden Nodes: ' num2str(hidden(h))]})
            ylabel('Time series x(t)')
            xlabel('Time step t')
            legend('Test targets','Test predictions')
            best_test = av_val_mse_matrix(r,h);
            
            figure
            hold on
            plot(linspace(1261, 1500, 240), e(961:1200));
            title({['Best Model']; ['Strength of Regularisation: ' num2str(reg_strength(r))]; ['Number of Hidden Nodes: ' num2str(hidden(h))]})
            ylabel('Error')
            xlabel('Time step t')
            legend('Test error')
        end
        
    end
    
    figure
    histogram(weights, 'BinWidth', 0.25)
    title({['Histogram of Weights']; ['Strength of Regularisation: ' num2str(reg_strength(r))]});
    xlabel('Weight value')
    ylabel('Counts')
end

av_val_mse_matrix
std_val_mse_matrix

% _______________________________4.3.2______________________________________ 
%% Three-layer perceptron for noisy time series predicition 

% Choose a Training Function (help nntrain)
trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation.

% Early stopping
%net.trainParam.max_fail = 6;    % Maximum validation failures

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';          % No shuffling, we want consecutive blocks
net.divideMode = 'sample';
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;

% Choose a Performance Function (help nnperformance)
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions (help nnplot)
net.plotFcns = {'plotperform', 'plotfit'};

% Configurations
reg_strength = [0.0, 0.25, 0.5, 0.75 , 1.0];
hidden = [2, 4, 6, 8];
% Noise parameters
sigma = [0.03, 0.09, 0.18];

% Calculate average mse and std for each configuration 
av_test_mse_matrix = zeros(length(reg_strength), length(hidden));
std_test_mse_matrix = zeros(length(reg_strength), length(hidden));

for s = 1:length(sigma)
    for r = 1:length(reg_strength)
        for h = 1:length(hidden) 
            % Set strength of regularization 
            net.performParam.regularization = reg_strength(r);
            % Set number of hidden nodes
            hiddenLayerSize = hidden(h);

            % Create a Fitting Network 
            net = fitnet([6,hiddenLayerSize],trainFcn);

            % Initialize mse for this configuration
            train_mse = [];
            val_mse = [];
            test_mse = [];

            % Repeat for valid statistics
            repeats = 100;
            for i = 1:repeats;
                % Initalize
                net = init(net);
                
                t_noise = t + normrnd(0, sigma(s), [1,n]);

                % Train the Network
                [net,tr] = train(net,x,t_noise);

                % Test the Network
                y = net(x);
                %e = gsubtract(t_noise,y);
                %performance = perform(net,t,y);
                nntraintool close;

                % Recalculate Test Performance
                testTargets = t_noise .* tr.testMask{1};
                testPerformance = perform(net,testTargets,y);
                test_mse = [test_mse testPerformance];

            end
            av_test_mse_matrix(r,h) = mean(test_mse);  
            std_test_mse_matrix(r,h) = std(test_mse);  
        end
    end
    
    sigma(s)
    av_test_mse_matrix
    std_test_mse_matrix
end

%% Continued - evaluate two-layer

% Choose a Training Function (help nntrain)
trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation.

% Early stopping
%net.trainParam.max_fail = 6;    % Maximum validation failures

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';          % No shuffling, we want consecutive blocks
net.divideMode = 'sample';
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;

% Choose a Performance Function (help nnperformance)
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions (help nnplot)
net.plotFcns = {'plotperform', 'plotfit'};

% Noise parameters
sigma = [0.03, 0.09, 0.18];

for s = 1:length(sigma)
    % Set strength of regularization 
    net.performParam.regularization = 0.00;
    % Set number of hidden nodes
    hiddenLayerSize = 6;

    % Create a Fitting Network 
    net = fitnet(hiddenLayerSize,trainFcn);

    % Initialize mse for this configuration
    test_mse = [];

    % Repeat for valid statistics
    repeats = 100;
    for i = 1:repeats
        % Initalize
        net = init(net);

        t_noise = t + normrnd(0, sigma(s), [1,n]);

        % Train the Network
        [net,tr] = train(net,x,t_noise);

        % Test the Network
        y = net(x);
        %e = gsubtract(t_noise,y);
        %performance = perform(net,t,y);
        nntraintool close;

        % Recalculate Test Performance
        testTargets = t_noise .* tr.testMask{1};
        testPerformance = perform(net,testTargets,y);
        test_mse = [test_mse testPerformance];
    end
    
    sigma(s)
    av_test_mse = mean(test_mse)
    std_test_mse = std(test_mse)
end

%% Evaluate convergence 

% Choose a Training Function (help nntrain)
trainFcn = 'trainscg'; % Scaled conjugate gradient backpropagation.

% Early stopping
%net.trainParam.max_fail = 6;    % Maximum validation failures

% Setup Division of Data for Training, Validation, Testing
net.divideFcn = 'divideblock';          % No shuffling, we want consecutive blocks
net.divideMode = 'sample';
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;

% Choose a Performance Function (help nnperformance)
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions (help nnplot)
net.plotFcns = {'plotperform', 'plotfit'};

% Configurations
reg_strength = [0.00];
hidden = [2, 4, 6, 8];
% Noise parameters
sigma = [0.03];

% Calculate average mse for each configuration 

for s = 1:length(sigma)
    for r = 1:length(reg_strength)
        for h = 1:length(hidden) 
            % Set strength of regularization 
            net.performParam.regularization = reg_strength(r);
            % Set number of hidden nodes
            hiddenLayerSize = hidden(h);

            % Create a Fitting Network 
            net = fitnet([6,hiddenLayerSize],trainFcn);

            % Repeat for valid statistics
            repeats = 100;
            cost = [];
            time = [];
            for i = 1:repeats;
                % Initalize
                net = init(net);
                
                t_noise = t + normrnd(0, sigma(s), [1,n]);

                % Train the Network
                time_i = cputime;
                [net,tr] = train(net,x,t_noise);
                time_e = cputime-time_i;
                nntraintool close;
                
                time = [time  time_e];
                cost = [cost tr.num_epochs];

            end
            h
            av_cost = mean(cost)
            std_cost = std(cost)
            
            av_time = mean(time)
            std_time = std(time)
        end
    end
end

%%

for s = 1:length(sigma)
    % Set strength of regularization 
    net.performParam.regularization = 0.00;
    % Set number of hidden nodes
    hiddenLayerSize = 6;

    % Create a Fitting Network 
    net = fitnet(hiddenLayerSize,trainFcn);

    % Repeat for valid statistics
    repeats = 100;
    cost = [];
    time = [];
    for i = 1:repeats;
        % Initalize
        net = init(net);

        t_noise = t + normrnd(0, sigma(s), [1,n]);

        % Train the Network
        time_i = cputime;
        [net,tr] = train(net,x,t_noise);
        time_e = cputime-time_i;
        nntraintool close;
        
        time = [time  time_e];
        cost = [cost tr.num_epochs];

    end
    
    av_cost = mean(cost)
    std_cost = std(cost)

    av_time = mean(time)
    std_time = std(time)

end


