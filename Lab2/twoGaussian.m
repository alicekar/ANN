function [fxy] = twoGaussian(x , y, meanx , meany , sigmax , sigmay)

fxy = exp(   -  (  ((x-meanx)^2 / 2*sigmax^2) + ((y-meany)^2 / 2*sigmay^2)  ));


end

