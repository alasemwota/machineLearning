function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = size(theta);

for i=1:m
    h_x = sigmoid(X(i,:)*theta);
    J += y(i)*log(h_x) + (1-y(i))*log(1-h_x);
end
J = (J / m) * -1;

regFactor = 0;
for k=2:n
    regFactor += theta(k)^2;
end
regFactor *= lambda / (2*m);
J += regFactor;

for h = 1:n
    for i = 1:m
        h_x = sigmoid(X(i,:)*theta);
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
