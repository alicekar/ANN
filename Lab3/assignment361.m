% Author: Irene

clc
clear all 

Teta = linspace(0,35,5);
%Teta = 0.10;

t = 1;
p = 1;
Perc = linspace(0 , 0.5 , 5);
%Perc = 0.10;
%perc = Perc
final_stored = zeros(length(Teta) , length(Perc));


for teta = Teta
    p = 1;
    for perc = Perc

        %Train Data creation - 300 100-dim random patterns 
        n_patterns = 300;
        N = 100;
        
        train = zeros(n_patterns, N);

        for i = 1:n_patterns
            idx = randi(N, 1, round(N*perc));
            
            train(i, idx ) = 1;
        end


        %Test Data creation
        test = train;

        % Average activity 
        ro = (1/ (N * n_patterns) ) * sum(sum(train));
        %Weights Train
        W = zeros(N,N);
        for i = 1 : n_patterns
            W = W + (train(i,:)' - ro) * ( train(i,:) - ro) ;
        end

        out = zeros (n_patterns, N);
        old_out = ones(n_patterns, N);
        termine = zeros(n_patterns,1);

        %Check if the inputs are stored 
        for i = 1:n_patterns
            termine(i) = j;
            j = 1;

            while  old_out(i,:)~= out(i,:)

                old_out(i,:) = out(i,:);

                if j == 1
                    out(i,:) = [0.5 + 0.5 * Sign(W * test(i,:)' - teta)'] ;
                else 
                    out(i,:) = [0.5 + 0.5 * Sign(W * out(i,:)' - teta)'] ;
                end
                j = j+1;
            end

        end

        %Compute errror 
        error = sum(abs(train'-out'))/2;
        [r , idx] = find(error == 0);
        
        final_stored(t,p) = length(idx);
        
        p = p+1;
        

    end
    
             
        t = t+1;
end

for t = 1:length(Teta)
    plot ( Perc , final_stored(t,:))
    hold on
end
legend('bias = 0 ' , 'bias = 8.750 ' , 'bias = 17.50 ' , 'bias = 26.250' , ' bias = 35 ')
xlabel('rho average activity')
ylabel('# stored patterns')













