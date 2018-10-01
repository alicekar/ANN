function [sum_errA, sum_errB] = getError2(W, X, T)
   Y = W*X;
   Y(Y>=0)=1; 
   Y(Y<0)=-1;
   check = [Y;T];
   Aidx = find(check(2,:) == 1);
   Apart = check(:,Aidx);
   Bidx = find(check(2,:) == -1);
   Bpart = check(:,Bidx);
   errorA = Apart(2,:)-Apart(1,:);  %T-Y
   errorB = Bpart(2,:)-Bpart(1,:);
   sum_errA = sum(abs(errorA))/2;
   sum_errB = sum(abs(errorB))/2;
end