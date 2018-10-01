% Author: Alice 

classdef ass_32
    methods (Static)
        % Import and organize data
        function data = import_data()
            data = importdata('pict.dat');
            data = reshape(data,1024,11);
            data = data';
        end
        
        % Train
        function W = get_W(train_data,N)
            W = train_data'*train_data*(1/N);
        end
        
        % Sequential update
        function output = seq_update(input,W,N,max_iter)
            output = input;
            for iter = 1:max_iter
                for i = 1:N
                    w_i = W(i,:);
                    inner_sum = sum(w_i.*output);
                    output(i) = sign(inner_sum);
                end
            end
        end
                
        % Check convergens, 0 means that the output reached correct
        % attractor
        function convergens = get_conv(input,output)
                n_nonzero = nnz(input-output);
                convergens = n_nonzero;
        end  
        
        % Check that the 3 patterns are stable
        function answer_Q1 = three_stable(train,W,N,max_iter) 
            answer_Q1 = zeros(1,size(train,1));
            for i = 1:size(train,1)
                input = train(i,:);
                output = ass_32.seq_update(input,W,N,max_iter);
                answer_Q1(i) = ass_32.get_conv(input,output);
            end
        end
        
        % Can the network complete a degraded pattern? Try the pattern p10, which
        % is a degraded version of p1, or p11 which is a mixture of p2 and p3.
        function answer_Q2 = complete_degraded(data, W, N, max_iter)
            p = [data(10,:); data(11,:); data(11,:)];
            attractors = [data(1,:); data(2,:); data(3,:)];
            answer_Q2 = zeros(1,size(p,1));
            outputs = zeros(size(p,1),N);
            steps = [0, 2, 4];
            figure;
            for i = 1:size(p,1)
                output = ass_32.seq_update(p(i,:),W,N,max_iter);
                attractor = attractors(i,:);
                answer_Q2(i) = ass_32.get_conv(attractor,output);
                outputs(i,:) = output;
                
                hold on;
                attractor = mat2gray(reshape(attractor,32,32));
                result_deg = mat2gray(reshape(output,32,32));
                start = mat2gray(reshape(p(i,:),32,32));
                subplot(3,3,i+steps(i));
                imshow(start,'InitialMagnification', 800)
                xlabel('Before');
                if i == 1
                    ylabel('p10 -> p1')
                elseif i == 2
                    ylabel('p11 -> p2')               
                elseif i == 3
                    ylabel('p11 -> p3')
                end
                subplot(3,3,i+steps(i)+1);
                imshow(result_deg,'InitialMagnification', 800)
                xlabel('After');
                subplot(3,3,i+steps(i)+2);
                imshow(attractor,'InitialMagnification', 800)
                xlabel('Attractor'); 
            end
        end
        
        % What happens if we select units randomly? Please calculate their 
        % new state and then repeat the process in the spirit of the original 
        % sequential Hopfield dynamics. Please demonstrate the image every 
        % hundredth iteration or so.
        function answer_Q3(input,attractor,W,N,max_iter)
            output = input;
            figure;
            hold on;
            for iter = 1:(N*max_iter)
                i = randi(N);
                w_i = W(i,:);
                inner_sum = sum(w_i.*output);
                output(i) = sign(inner_sum);
                
                if iter==500
                    result = mat2gray(reshape(output,32,32));
                    subplot(2,2,1);
                    imshow(result,'InitialMagnification', 1000);
                    xlabel('500 iterations');
                elseif iter==1000
                    result = mat2gray(reshape(output,32,32));
                    subplot(2,2,2);
                    imshow(result,'InitialMagnification', 1000);
                    xlabel('1000 iterations');
                elseif iter==3000
                    result = mat2gray(reshape(output,32,32));
                    subplot(2,2,3);
                    imshow(result,'InitialMagnification', 1000);
                    xlabel('3000 iterations');
                elseif iter==5500
                    result = mat2gray(reshape(output,32,32));
                    subplot(2,2,4);
                    imshow(result,'InitialMagnification', 1000);
                    xlabel('5500 iterations');
                end
            end
            figure;
            attr = mat2gray(reshape(attractor,32,32));
            imshow(attr,'InitialMagnification', 1000);
        end
        
        % Run algorithm
        function run()
            data = ass_32.import_data();
            train = data(1:3,:);
            N = size(data,2);
            P = size(train,1); % Number of training patterns
            max_iter = round(log(N));
            W = ass_32.get_W(train,N);
            %answer_Q1 = ass_32.three_stable(train,W,N,max_iter)
            %answer_Q2 = ass_32.complete_degraded(data, W, N, max_iter)
            input = data(11,:);
            attractor = train(2,:);
            ass_32.answer_Q3(input,attractor,W,N,max_iter)
        end
    end
end