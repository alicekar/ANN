% Author: Alice 

classdef ass_33
    methods (Static)
        % Import and organize data
        function data = import_data()
            data = importdata('pict.dat');
            data = reshape(data,1024,11);
            data = data';
        end
        
        % Train
        function W = get_W(train_data,N)
            % Symmetric matrix
            W = train_data'*train_data*(1/N);
        end
        
        % Random W
        function W_rand = get_rand_W(N)
            W_rand = normrnd(0,0.5,[N,N]);
        end    
        
        function W_randSym = get_randSym_W(N)
            W_rand = ass_33.get_rand_W(N);
            W_randSym = 0.5*(W_rand+W_rand');
        end
            
        % Energy function
        function E = energy(W,pattern)
            x = pattern;
            E = 0;
            for i = 1:size(W,1)
                for j = 1:size(W,2)
                    E = E+(W(i,j)*x(i)*x(j));
                end
            end
            E = -E;
        end
        
        % What is the energy at the different attractors?
        function answer_Q1 = attractors_E(attractors,W)
            answer_Q1 = zeros(1,size(attractors,1));
            for i = 1:size(attractors,1)
               E_attractor = ass_33.energy(W,attractors(i,:));
               answer_Q1(i) = E_attractor;
            end    
        end
        
        % What is the energy at the points of the distorted patterns?
        function answer_Q2 = distorted_E(W,distorted_patterns)
            answer_Q2 = zeros(1,size(distorted_patterns,1));
            for i = 1:size(distorted_patterns,1)
               E_distorted = ass_33.energy(W,distorted_patterns(i,:));
               answer_Q2(i) = E_distorted;
            end    
        end
        
        % Sequential update, choose units randomly
        function all_E = E_randSeq(input,W,N,max_iter)
            all_E = zeros(1,max_iter/100);
            output = input;
            for iter = 1:max_iter
                i = randi(N);
                w_i = W(i,:);
                inner_sum = sum(w_i.*output);
                output(i) = sign(inner_sum);
                if mod(iter,100) == 0
                   idx = iter/100;
                   E = ass_33.energy(W,output);
                   all_E(idx) = E;
                end    
            end
        end
        
        % Follow how the energy changes from iteration to iteration when 
        % you use the sequential update rule to approach an attractor.
        function answer_Q3(iter_result, max_iter, dist_patterns, W, N)
            p10 = dist_patterns(1,:);
            p11 = dist_patterns(2,:);
            E_matrix_10 = zeros(iter_result,max_iter/100);
            E_matrix_11 = zeros(iter_result,max_iter/100);
            for i = 1:iter_result
               E_matrix_10(i,:) = ass_33.E_randSeq(p10,W,N,max_iter); 
               E_matrix_11(i,:) = ass_33.E_randSeq(p11,W,N,max_iter); 
            end 
            mean_E_10 = mean(E_matrix_10)
            mean_E_11 = mean(E_matrix_11);
            std_E_10 = std(E_matrix_10);
            std_E_11 = std(E_matrix_11);
            x_axis = 100*(1:size(mean_E_10,2));
            
            figure;
            hold on;
            errorbar(x_axis,mean_E_10,std_E_10,'r')
            errorbar(x_axis,mean_E_11,std_E_11,'b')
            legend('p10','p11')
            title('Decrease in Energy over Iterations')
            xlabel('Number of iterations')
            ylabel('Energy')
        end
        
        % Generate a weight matrix by setting the weights to normally distr-
        % ibuted random numbers, and try iterating an arbitrary starting state.
        function answer_Q45(iter_result, max_iter, dist_patterns, W, N)
            p8 = dist_patterns(1,:);
            p9 = dist_patterns(2,:);
            E_matrix_8 = zeros(iter_result,max_iter/100);
            E_matrix_9 = zeros(iter_result,max_iter/100);
            for i = 1:iter_result
               E_matrix_8(i,:) = ass_33.E_randSeq(p8,W,N,max_iter); 
               E_matrix_9(i,:) = ass_33.E_randSeq(p9,W,N,max_iter); 
            end 
            mean_E_8 = mean(E_matrix_8);
            mean_E_9 = mean(E_matrix_9);
            std_E_8 = std(E_matrix_8);
            std_E_9 = std(E_matrix_9);
            x_axis = 100*(1:size(mean_E_8,2));
            
            figure;
            hold on;
            errorbar(x_axis,mean_E_8,std_E_8,'g')
            errorbar(x_axis,mean_E_9,std_E_9,'m')
            legend('p8','p9')
            title('Decrease in Energy over Iterations')
            xlabel('Number of iterations')
            ylabel('Energy')
        end
        
        % Run algorithm
        function run()
            % Data and constants
            data = ass_33.import_data();
            train = data(1:3,:);
            N = size(data,2);
            
            % Distorted patterns of p1, p2 and p3
            distorted_patterns = data(10:11,:);
            
            % Arbitrary states, Q4 and Q5
            arb_patterns = data(8:9,:);
            
            % Iterations 
            max_iter = 7500;   % Differs from ass_32
            iter_result = 10;  % To avergage over for the plots
            
            % Different W's
            W = ass_33.get_W(train,N);
            rand_W = ass_33.get_rand_W(N);
            W_randSym = ass_33.get_randSym_W(N);
            
            disp('runs')
            
            % Answers
            answer_Q1 = ass_33.attractors_E(train,W)
            answer_Q2 = ass_33.distorted_E(W,distorted_patterns)
            %ass_33.answer_Q3(iter_result, max_iter, distorted_patterns, W, N)
            % Answer Q4
            %ass_33.answer_Q45(iter_result, max_iter, arb_patterns, rand_W, N)
            % Answer Q5
            %ass_33.answer_Q45(iter_result, max_iter, arb_patterns, W_randSym, N)
            
        end
    end
end