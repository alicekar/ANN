% Author: Alice

classdef assignment_43
    methods (Static)
        % Read in the cities data
        function [votes, party, gender, district] = read_votes() 
            votes = importdata('data_lab2/votes.dat');
            votes = reshape(votes, 31, 349); 
            votes = votes';
            % 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
            party = importdata('data_lab2/mpparty.dat'); % float, 
            gender = importdata('data_lab2/mpsex.dat'); % float
            gender = gender(:,2); % 0=male, 1=female
            district = importdata('data_lab2/mpdistrict.dat'); % float, 29
        end    
        
        % Initial values for the weights
        function W = initial_W(dim_in, dim_out)
            W = rand(dim_out, dim_in ,dim_out);
        end
        
        % Calc the similarity between the input pattern and the weights
        % arriving at each output node
        function similarity = distance(x,W)
            diff = sum((W-x).^2,2);
            similarity = reshape(diff,size(diff,1),size(diff,3));
        end 
        
        % Get row and col indices from single matrix index
        function [row, col] = row_col(index, dim_out)
            if index <= dim_out
                col = 1;
                row = index;   
            elseif (index > dim_out) && (mod(index,dim_out) == 0)
                col = index/dim_out;
                row = 10;
            else
                col = floor(index/dim_out)+1;
                row = mod(index, dim_out);
            end
        end
        
        % Find node closes to input
        function winner = winning_node(x,W)
            similarity = assignment_43.distance(x,W);
            flatten = reshape(similarity',1, size(similarity,1)*size(similarity,2));
            minimum = min(flatten);
            % single index in matrix goes [1 4 7; 2 5 8; 3 6 9] etc
            winner = find(similarity == minimum); 
        end 
        
        % Calc neighbourhoods sizes dep on N_epochs
        function neighbourhood_sizes = neighbourhoods(N_epochs)
            use_sizes = [5 4 3 2 2 1 1 1 0 0];
            %use_sizes = [5 2];
            amount = floor(N_epochs/length(use_sizes));
            final_sizes = zeros(1, N_epochs);
            start = 1;
            for i = 1:length(use_sizes)
                final_sizes(1,start:start+amount) = use_sizes(i);
                start = start+amount;
            end
            neighbourhood_sizes = final_sizes  
        end
        
        % Find neighbours
        % In the two-dimensional case, is is normally sufficient to use 
        % Manhattan distance, i.e. to add the absolute values of the index 
        % differences in row and column directions.
        function neighbours = neighbours(winner, neigh_size, dim_out)
            [x, y] = assignment_43.row_col(winner, dim_out);
            neighbours_x = [];
            neighbours_y = [];
            for i = (x-neigh_size):(x+neigh_size)
                steps = abs(abs(x-i)-neigh_size);
                for j = (y-steps):(y+steps)
                    if (i>0 && j>0 && i<=dim_out && j<=dim_out)
                        neighbours_x = [neighbours_x i];
                        neighbours_y = [neighbours_y j];
                    end
                end
            end   
            neighbours = [neighbours_x; neighbours_y];
        end
         
        % Update weights for neighbour nodes
        function new_xW = update_neighbours_W(W, neighbours, eta, x)
            for i = 1:size(neighbours,2)
                neighbour = neighbours(:,i);
                neigh_x = neighbour(1);
                neigh_y = neighbour(2);
                w_i = W(neigh_x,:,neigh_y);
                w_i_new = w_i+eta.*(x-w_i);
                W(neigh_x,:,neigh_y) = w_i_new;
            end    
            new_xW = W;    
        end
        
        % Update W depending on all x's and for a number of epochs
        function pos_coordinates = SOM_algo(N_epochs, eta, start_W, X, dim_out)
            neighbourhood_sizes = assignment_43.neighbourhoods(N_epochs);
            W = start_W;
            for epoch = 1:N_epochs
                neigh_size = neighbourhood_sizes(epoch);
                for i = 1:size(X,1)
                    x = X(i,:);
                    winner = assignment_43.winning_node(x,W);
                    neighbours = assignment_43.neighbours(winner, neigh_size, dim_out);
                    new_xW = assignment_43.update_neighbours_W(W, neighbours, eta, x);
                    W = new_xW;
                end
            end
            new_W = W;
            
            all_last_winners = zeros(1,size(X,1));
            for i = 1:size(X,1)
                x = X(i,:);
                winner = assignment_43.winning_node(x,new_W);
                all_last_winners(i) = winner;
            end
            pos = all_last_winners;

            % Get pos in form of coordinates in the grid
            pos_coordinates = zeros(2,length(pos));
            for i = 1:length(pos)
                p = pos(i);
                [x_p,y_p] = assignment_43.row_col(p, dim_out);
                pos_coordinates(1,i) = x_p;
                pos_coordinates(2,i) = y_p;
            end
        end
        
        % Plot solution
        function plot_votes(pos_coordinates, gender, party, district)
            pos_coord_noise = zeros(2,size(gender,1));
            mean = 0;
            std = 0.33;
            for i = 1:size(gender,1)
                x = pos_coordinates(1,i) + normrnd(mean,std);
                y = pos_coordinates(2,i) + normrnd(mean,std);
                pos_coord_noise(1,i) = x;
                pos_coord_noise(2,i) = y;
            end    
            %{
            % Gender plot
            gender_coord = [pos_coord_noise; gender'];
            figure(1)
            hold on
            for i = 1:size(gender,1)
                x = gender_coord(1,i);
                y = gender_coord(2,i);
                if gender_coord(3,i)==0
                    plot(x,y,'b*')
                else
                    plot(x,y,'r*')
                end
            end
            h = zeros(2, 1);
            h(1) = plot(NaN,NaN,'b*');
            h(2) = plot(NaN,NaN,'r*');
            legend(h, 'male','female');
            title('Gender');
            
            % Party plot
            % 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
            party_coord = [pos_coord_noise; party'];
            figure(2)
            hold on
            for i = 1:size(party,1)
                x = party_coord(1,i);
                y = party_coord(2,i);
                if party_coord(3,i)==0
                    plot(x,y,'k*')
                elseif party_coord(3,i)==1
                    plot(x,y,'b*')
                elseif party_coord(3,i)==2
                    plot(x,y,'co')
                elseif party_coord(3,i)==3
                    plot(x,y,'r*')
                elseif party_coord(3,i)==4
                    plot(x,y,'ro')
                elseif party_coord(3,i)==5
                    plot(x,y,'g*')
                elseif party_coord(3,i)==6
                    plot(x,y,'m*')
                else
                    plot(x,y,'y*')
                end
            end
            h = zeros(8, 1);
            h(1) = plot(NaN,NaN,'k*');
            h(2) = plot(NaN,NaN,'b*');
            h(3) = plot(NaN,NaN,'co');
            h(4) = plot(NaN,NaN,'r*');
            h(5) = plot(NaN,NaN,'ro');
            h(6) = plot(NaN,NaN,'g*');
            h(7) = plot(NaN,NaN,'m*');
            h(8) = plot(NaN,NaN,'y*');
            legend(h, 'no party','m','fp','s','v','mp','kd','c');
            title('Party')
            %}
            % District
            district_coord = [pos_coord_noise; district'];
            figure(2)
            hold on
            for i = 1:size(district,1)
                x = district_coord(1,i);
                y = district_coord(2,i);
                if district_coord(3,i)==1
                    plot(x,y,'b*')
                elseif district_coord(3,i)==2
                    plot(x,y,'c*')
                elseif district_coord(3,i)==3
                    plot(x,y,'r*')
                elseif district_coord(3,i)==4
                    plot(x,y,'k+')
                elseif district_coord(3,i)==5
                    plot(x,y,'g*')
                elseif district_coord(3,i)==6
                    plot(x,y,'m*')
                elseif district_coord(3,i)==7
                    plot(x,y,'y*')
                elseif district_coord(3,i)==8
                    plot(x,y,'k*')
                elseif district_coord(3,i)==9
                    plot(x,y,'bo')
                elseif district_coord(3,i)==10
                    plot(x,y,'co')
                elseif district_coord(3,i)==11
                    plot(x,y,'r+')
                elseif district_coord(3,i)==12
                    plot(x,y,'ro')
                elseif district_coord(3,i)==13
                    plot(x,y,'go')
                elseif district_coord(3,i)==14
                    plot(x,y,'mo')
                elseif district_coord(3,i)==15
                    plot(x,y,'yo')
                elseif district_coord(3,i)==16
                    plot(x,y,'ko')
                elseif district_coord(3,i)==17
                    plot(x,y,'b>')
                elseif district_coord(3,i)==18
                    plot(x,y,'c>')
                elseif district_coord(3,i)==19
                    plot(x,y,'r>')
                elseif district_coord(3,i)==20
                    plot(x,y,'b+')
                elseif district_coord(3,i)==21
                    plot(x,y,'g>')
                elseif district_coord(3,i)==22
                    plot(x,y,'m>')
                elseif district_coord(3,i)==23
                    plot(x,y,'y>')
                elseif district_coord(3,i)==24
                    plot(x,y,'k>')
                elseif district_coord(3,i)==25
                    plot(x,y,'cd')
                elseif district_coord(3,i)==26
                    plot(x,y,'rd')
                elseif district_coord(3,i)==27
                    plot(x,y,'rd')
                elseif district_coord(3,i)==28
                    plot(x,y,'gd')
                else 
                    plot(x,y,'m*')
                end
            end
            title('Districts')
            
        end
        
        % Get results for vote exercise
        function run_votes(dim_out, N_epochs, eta)
            [votes, party, gender, district] = assignment_43.read_votes();
            dim_in = size(votes, 2);
            start_W = assignment_43.initial_W(dim_in, dim_out);
            pos_coordinates = assignment_43.SOM_algo(N_epochs, eta, start_W, votes, dim_out);
            assignment_43.plot_votes(pos_coordinates, gender, party, district);
            disp('KLART')
        end
    end
end