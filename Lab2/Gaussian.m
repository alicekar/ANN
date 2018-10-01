function [y] = Gaussian(x , mu , sigma)

y = exp(-(x-mu).^2 / 2*sigma.^2);

end

