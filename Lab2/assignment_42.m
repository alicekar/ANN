% Author: Alice

classdef assignment_42
    methods (Static)
        % Read in the cities data
        function coordinates = read_cities() 
             data = importdata('data_lab2/cities.dat');
             data = data(3:end,:);
             data = char(data);
             coordinates = str2num(data);   % float
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
            similarity = assignment_42.distance(x,W);
            minimum = min(similarity);
            winner = find(similarity == minimum);
        end 
        
        % Make the size of the neighbourhood depend on the epoch loop var
        function neighbourhood_sizes_cities = neighbourhood_size_cities(N_epochs)
            use_sizes = [2 2 1 1 1 1 1 0 0 0];
            amount = floor(N_epochs/length(use_sizes));
            final_sizes = zeros(1, N_epochs);
            start = 1;
            for i = 1:10
                final_sizes(1,start:start+amount) = use_sizes(i);
                start = start+amount;
            end
            neighbourhood_sizes_cities = final_sizes;   
        end

        % Find neighbours in cyclic node space 
        function neighbours = neighbours_cyclic(winner, size, dim_outputs)
            first = winner-size;
            last = winner+size;
            neighbours = first:1:last;
            if first == 0
                neighbours = [dim_outputs, 1:1:last];
            end
            if first == -1
                neighbours = [dim_outputs-1 dim_outputs, 1:1:last];
            end
            if last == dim_outputs+1
                neighbours = [first:1:dim_outputs 1];
            end
            if last == dim_outputs+2
                neighbours = [first:1:dim_outputs 1 2];
            end
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
        function pos = SOM_algo(N_epochs, eta, start_W, X, dim_outputs)
            neighbourhood_sizes = assignment_42.neighbourhood_size_cities(N_epochs);
            W = start_W;
            for epoch = 1:N_epochs
                neighbourhood_size = neighbourhood_sizes(epoch);
                for i = 1:size(X,1)
                    x = X(i,:);
                    winner = assignment_42.winning_node(x,W);
                    neighbours = assignment_42.neighbours_cyclic(winner, neighbourhood_size, dim_outputs);
                    new_xW = assignment_42.update_neighbours_W(W, neighbours, eta, x);
                    W = new_xW;
                end
            end
            new_W = W;
            all_last_winners = zeros(1,size(X,1));
            for i = 1:size(X,1)
                x = X(i,:);
                winner = assignment_42.winning_node(x,new_W);
                all_last_winners(i) = winner;
            end
            pos = all_last_winners;
        end
        
        % Sorts pos into increasing order = ~ 
        % finds the indexes of ~ in pos = ordered_idx
        function ordered_idx = order(pos)
            [~, ordered_idx] = mink(pos, length(pos));
        end
        
        % Plot the solution
        function plot_tour(X, pos)
            ordered_idx = assignment_42.order(pos);
            ordered_coordinates = X(ordered_idx, :);
            hold on 
            axis([0 1 0 1])
            plot(ordered_coordinates(:,1),ordered_coordinates(:,2))
            plot(ordered_coordinates(:,1),ordered_coordinates(:,2),'r*')
            title('Tour Plot')
            xlabel('x_1')
            ylabel('x_2')
        end
        
        % Get results for cyclic tour exercise
        function run_cities(N_epochs, eta, dim_outputs)
            coordinates = assignment_42.read_cities();
            dim_in = size(coordinates,2);
            start_W = assignment_42.initial_W(dim_in, dim_outputs);
            shuffled_idx = randperm(size(coordinates,1));
            X = coordinates(shuffled_idx,:);
            pos = assignment_42.SOM_algo(N_epochs, eta, start_W, X, dim_outputs); 
            assignment_42.plot_tour(X, pos)
        end
        
    end
end