function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);
J = 0;
grad = zeros(size(theta));

J = -(1/m) * (transpose(y) * log(sigmoid(X * theta)) ...
  + transpose((1 - y)) * log(1 - sigmoid(X * theta))) ... 
  + lambda / (2 * m) * (sum(theta .^ 2)-(theta(1)^2));

grad = (1/m) * transpose(X) * (sigmoid(X * theta) - y) + (lambda / m) * theta;
grad(1) = (1/m) * transpose(sigmoid(X * theta) - y) * X(:, 1);

end