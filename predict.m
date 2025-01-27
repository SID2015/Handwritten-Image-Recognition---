function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% we will return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: The following code will make predictions using
%               our learned neural network. we should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element. 
%        If our examples are in rows, then, we
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];

z1 = (X)*(Theta1');

z2 = sigmoid(z1);

k = [ones(m, 1) z2];

z3 = (k)*(Theta2');

[x,p] = max(z3,[],2);


% =========================================================================


end
