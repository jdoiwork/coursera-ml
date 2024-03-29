function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% y
% size(y) % => 5000, 1
% input("size of y");

% X
% size(X) % => 5000, 400
% input("size of x"); 

% size(Theta1) % => 25, 401
% input("size of theta1"); 

% size(Theta2) % => 10, 26
% input("size of theta2"); 

% num_labels
% input("num_labels"); % => 10

s = 0;

for t = 1:m
  x = X(t, :);
  yi = y(t);

  vy = labelAsVector(yi, num_labels);

  % Step 1
  [a1, z2, a2, z3, a3] = myCostStep(vy, x', Theta1, Theta2);
  s += myCost(a3, vy);



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%



  % Step 2
  delta3 = a3 - vy';

  % Step 3
  sg2 = [1; sigmoidGradient(z2)];
  delta2 = (Theta2' * delta3) .* sg2;

  % Step 4

  Theta1_grad += delta2(2:end) * a1';
  Theta2_grad += delta3 * a2';
  
endfor

r = regularizedThetas(Theta1, Theta2);

J = (s / m) + (lambda / (2 * m) * r);

Theta1_grad /= m;
Theta2_grad /= m;




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



reg1 = (lambda / m) * Theta1;
reg2 = (lambda / m) * Theta2;

% a = size(reg1(:, 1))
% b = size(zeros(size(reg1, 1), 1))

% input("reg1")

reg1(:, 1) = zeros(size(reg1, 1), 1);
reg2(:, 1) = zeros(size(reg2, 1), 1);

Theta1_grad += reg1;
Theta2_grad += reg2;















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
