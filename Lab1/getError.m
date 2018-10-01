function sum_err = getError(W, X, T)
   Y = W*X;
   Y(Y>=0)=1; 
   Y(Y<0)=-1;
   error = Y-T;
   sum_err = sum(abs(error))/2;
end

