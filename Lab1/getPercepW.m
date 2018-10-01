function [delta_W, sum_err] = getPercepW(W_old, eta, X, T)
X_transpose = X.';
Y = W_old*X;
Y(Y>=0)=1;
Y(Y<0)=-1;
delta_W = -eta*(Y-T)*X_transpose; 
end
