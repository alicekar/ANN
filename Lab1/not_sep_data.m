function [shuffled_classes,shuffled_targets] = not_sep_data(n1)
% Generate data
n = n1;
dim_outputs = 1;
mA = [ 1.0, 0.5]; sigmaA = 0.8;
mB = [-1.0, -0.1]; sigmaB = 0.8;

% Data
classA(1,:) = randn(1,n).*sigmaA + mA(1);
classA(2,:) = randn(1,n).*sigmaA + mA(2);
classB(1,:) = randn(1,n).*sigmaB + mB(1);
classB(2,:) = randn(1,n).*sigmaB + mB(2);
targetsA = ones(1,n);
targetsB = ones(1,n)*-1;
both_classes = [classA classB];
both_targets = [targetsA targetsB];
%SHUFFLE 
N = size(both_targets,2);
indices = randperm(N);
shuffled_classes = both_classes(:,indices);
shuffled_targets = both_targets(:,indices);

%hold on
%plot(classA(1,:), classA(2,:), '*')
%plot(classB(1,:), classB(2,:), '*')
end

