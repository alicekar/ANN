%% 4.1 Topological Ordering of Animal Species

% Author: Alice

clear all
import assignment_4.*
N_epochs = 20;
eta = 0.2;
dim_outputs = 100;
animal_result = assignment_4.run_animal(N_epochs, eta, dim_outputs)

%% 4.2 Cyclic Tour
clear all
close all
import assignment_42.*
N_epochs = 20;
eta = 0.2;
dim_outputs = 10;
assignment_42.run_cities(N_epochs, eta, dim_outputs);

%% 4.3 Data Clustering: Votes of MPs
clear all
close all
import assignment_43.*
dim_out = 10;
N_epochs = 20;
eta = 0.2;
assignment_43.run_votes(dim_out, N_epochs, eta)
