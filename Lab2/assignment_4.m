% Author: Alice

classdef assignment_4
    methods (Static)
        % Read in the animal data
        function [names, props] = read_animals() 
            props = importdata('data_lab2/animals.dat');
            props = reshape(props, 84, 32); 
            props = props';   
            names = importdata('data_lab2/animalnames.txt');
            names = string(names);
            names = strip(names);
            names = strip(names, "'");
        end    
        
        % Initial values for the weights
        function W = initial_W(dim_in, dim_out)
            W = rand(dim_out, dim_in);
        end
        
        % Calc the similarity between the input pattern and the weights
        % arriving at each output node
        function similarity = distance(x,W)
            similarity = sum((W-x).^2,2);
        end
        
        % Find node closes to input
        function winner = winning_node(x,W)
            similarity = assignment_4.distance(x,W);
            minimum = min(similarity);
            winner = find(similarity == minimum);
        end 
        
        % Make the size of the neighbourhood depend on the epoch loop var
        function neighbourhood_sizes_animal = neighbourhood_size_animal(N_epochs)
            use_sizes = [25 19 16 9 4 3 2 1 1 0];
            amount = floor(N_epochs/length(use_sizes));
            final_sizes = zeros(1, N_epochs);
            start = 1;
            for i = 1:10
                final_sizes(1,start:start+amount) = use_sizes(i);
                start = start+amount;
            end
            neighbourhood_sizes_animal = final_sizes;
        end
        
        % Find neighbours in linear node space 
        function neighbours = neighbours_1D(winner, size, dim_outputs)
            first = winner-size;
            last = winner+size;
            if first < 1
                % Must always be #2 x size" neighbours
                last = last-first+1; 
                first = 1;
            end
            if last > dim_outputs
                diff = last-dim_outputs; 
                first = first-diff;
                last = dim_outputs;
            end
            neighbours = first:1:last;
        end
        
        % Update weights for neighbour nodes
        function new_xW = update_neighbours_W(W, neighbours, eta, x)
            for i = neighbours
                w_i = W(i,:);
                w_i_new = w_i+eta.*(x-w_i);
                W(i,:) = w_i_new;
            end
            new_xW = W;    
        end
        
        % Update W depending on all x's and for a number of epochs
        function [new_W, pos] = SOM_algo(N_epochs, eta, start_W, X, dim_outputs)
            neighbourhood_sizes = assignment_4.neighbourhood_size_animal(N_epochs);
            W = start_W;
            for epoch = 1:N_epochs
                neighbourhood_size = neighbourhood_sizes(epoch);
                for i = 1:size(X,1)
                    x = X(i,:);
                    winner = assignment_4.winning_node(x,W);
                    neighbours = assignment_4.neighbours_1D(winner, neighbourhood_size, dim_outputs);
                    new_xW = assignment_4.update_neighbours_W(W, neighbours, eta, x);
                    W = new_xW;
                end
            end
            new_W = W;
            all_last_winners = zeros(1,size(X,1));
            for i = 1:size(X,1)
                x = X(i,:);
                winner = assignment_4.winning_node(x,new_W);
                all_last_winners(i) = winner;
            end
            pos = all_last_winners;
        end
        
        % Sort the resulting animal vector
        function animal_result = sort_animals(pos, T, dim_outputs)
            order = strings(1,dim_outputs);
            for i = 1:length(pos)
                animal = T(i);
                position = pos(i);
                order(position) = animal;
            end
            order(strcmp('',order)) = [];
            animal_result = order;
        end
            
        % Get result for animal exercise
        function result_animal = run_animal(N_epochs, eta, dim_outputs)
            [names, props] = assignment_4.read_animals();
            shuffled_idx = randperm(size(props,1));
            X = props(shuffled_idx,:);
            T = names(shuffled_idx,:);
            dim_inputs = size(props,2);
            start_W = assignment_4.initial_W(dim_inputs, dim_outputs);
            [new_w, pos] = assignment_4.SOM_algo(N_epochs,eta,start_W,X,dim_outputs)
            if length(pos) == length(unique(pos))
                disp('Results converged')
            else
                disp('Results did not converge')
            end
            result_animal = assignment_4.sort_animals(pos, T,dim_outputs);
        end    
    end
end