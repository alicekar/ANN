function [ y ] = Sign( x )

n = length(x);
y = zeros(n,1);

for i = 1:n
    if x(i) == 0 || x(i)>0
        y(i) = +1;
    else
        y(i) = -1;
end

end

