function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
err = zeros(1, size(val, 2)^2);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for i=1:size(val, 2)
	C = val(1, i);
	for j=1:size(val, 2)
		sigma = val(1, j);
		predictions = svmPredict(svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)), Xval);
		err(1, j + ((i-1) * size(val, 2))) = mean(double(predictions ~= yval));
	end
end

[m, m_ind] = min(err);
C = val(1, floor(m_ind/size(val, 2)) + 1);
sigma = val(1, mod(m_ind, size(val, 2)));





% =========================================================================

end
