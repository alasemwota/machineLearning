function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
for i = 1:m
    h_x = X(i,:)*theta;
    J += (h_x - y(i))^2;
end

J = J / (2*m);

regFactor = 0;
for k=2:n
    regFactor += theta(k)^2;
end
regFactor *= lambda / (2*m);
J += regFactor;

for h = 1:n
    for i = 1:m
        h_x = X(i,:)*theta;
        grad(h) += (h_x - y(i)) * X(i,h);        
    end
    grad(h) =  (grad(h) / m);
    if(h != 1)
        grad(h) += (theta(h)*lambda) / m;
    endif
end
% =============================================================

grad = grad(:);

end
