function delta_W = getDeltaW(W_old, eta, X, T)
X_transpose = X.';
Y = W_old*X;
delta_W = -eta*(Y-T)*X_transpose; 
end

