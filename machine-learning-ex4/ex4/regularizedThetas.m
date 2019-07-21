function ret = regularizedThetas(theta1, theta2)

% size(theta1)
% size(theta2)
% input("size of thetas");

sumsum = @(X) sum(sum(X(:, 2:end) .^ 2));

ret = sumsum(theta1) + sumsum(theta2);

end
