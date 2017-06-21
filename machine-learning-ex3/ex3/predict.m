function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X]; %adds a column of ones to X

int1 = X * Theta1'; %intermetdiate product 1
int1 = sigmoid(int1);
int1 = [ones(m, 1) int1]; %add a columns of one for the bias unit

int2 = int1 * Theta2'; %intermediate product 2
int2 = sigmoid(int2);
[maximum, p] = max(int2, [], 2); %store the maximum from each row in a column vector maxmimum and the index in another column vector p


% =========================================================================


end
