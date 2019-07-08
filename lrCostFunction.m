function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initializing some useful values
m = length(y); % number of training examples

% We will return the following variables as cost 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
%               Compute the cost of a particular choice of theta.             
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + OUR_CODE (using the temp variable)
%


h = X*theta;

h_sigm = sigmoid(h);

left_summation = ((y)')*(log(h_sigm));

right_summation = ((1-y)')*(log(1-h_sigm));

summation_cost = left_summation + right_summation ;

summation_cost = -((1/m)*(summation_cost));

temp= theta;
theta(1) = 0;
regularization = (lambda/(2*m))*((theta')*(theta));


J = summation_cost + regularization ;

temp(1) = 0;
grad = ((1/m)*(X')*(h_sigm - y)) + (lambda/m)*(temp);


% =============================================================

grad = grad(:);

end
