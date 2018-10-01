% Author: Irene

clc
clear all 

%Train Data creation
train = zeros(3,8);
train(1,:) = [-1 -1 1 -1 1 -1 -1 1];
train(2,:) = [-1 -1 -1 -1 -1 1 -1 -1];
train(3,:) = [-1 1 1 -1 -1 1 -1 1];

N = size(train,2);
n_patterns = size(train , 1);


%Test Data creation
set = [ones(1,N) , -ones(1,N)];
test = zeros(1,8);
test(1,:) = [1 -1 1 -1 1 -1 -1 -1];
test(2, :) = [1 1 -1 -1 -1 1 -1 -1];
test(3,:) = [1 1 1 -1 1 1 -1 1 ];

%testhalf(1,:) = [-train(1,1:4) , train(1,5:8)];
%testhalf(2, :) = [-train(2,1:4) , train(2,5:8)];
%testhalf(3,:) = [-train(3,1:4) , train(3,5:8)];
pp = test;
% test = nchoosek(set,N);
% p = unique(test,'rows');
% 
% pp = [];
% for i=1:size(p,1)
%     
%     pi = perms(p(i,:));
%     pi2 = unique(pi,'rows');
%     pp = [ pp ; pi2];
%     
% end

%Now pp is the set of vectors on which I need to test the nerwork to see
%which are the fixed points 


%Weights Train
W = zeros(N,N);
for i = 1 : n_patterns
    W = W + (1/N) * train(i,:)'*train(i,:);
end

%check if we stored the train patterns correcty / we should receive them as
%output 
% out1 = sign(W * train(1,:)');
% out2 = sign(W * train(2,:)');
% out3 = sign(W * train(3,:)');
%OK!



%Testing on pp test dataset 
p1 = size(pp,1);
p2 = size(pp,2);


out = zeros(p1,p2);
old_out = ones(p1,p2);

%termine = zeros(p1,1);

for i = 1:p1
    %termine(i) = j;
    j = 1;
    %for j = 1:100
    while  old_out(i,:)~= out(i,:)
        
        old_out(i,:) = out(i,:);
        
        if j == 1
            out(i,:) = [Sign(W * pp(i,:)')'] ;
        else 
            out(i,:) = [Sign(W * out(i,:)')'] ;
        end
        j = j+1;
    end
    
end
%Save unique rows 
%Uniq = unique(out, 'rows');
Uniq = out;
%Compute errror 
error = sum(abs(train'-Uniq'))/2;








