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

for rowIndex = 1:m
  x = X(rowIndex, :);
  % size(x) % => 1, 400
  % input("size of x");
  % y
  % input("y");
  %a3 = cost(y(rowIndex), x', Theta1, Theta2);
  % a3
  % size(a3) % => 10, 1
  % input("a3"); 

  yi = y(rowIndex);

  vy = labelAsVector(yi, num_labels);

  a3 = cost(vy, x', Theta1, Theta2);
  s += a3;

endfor

r = regularizedThetas(Theta1, Theta2);

J = (s / m) + (lambda / (2 * m) * r);

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

Delta = zeros(3, 1);

for t = 1:m
  x = X(t, :);
  yi = y(rowIndex);

  vy = labelAsVector(yi, num_labels);

  % Step 1
  [a1, z2, a2, z3, a3] = costStep(vy, x', Theta1, Theta2);

  % a1 = x'
  % size_a3 = size(a3)
  % size_vy = size(vy)

  % Step 2
  delta3 = a3 - vy';
  % delta3
  % input("delta 3")
  % size_delta3 = size(delta3)
  % input("size a3");

  % Step 3
  % size(Theta2')
  % sg2 = sigmoidGradient(z2);
  sg2 = [1; sigmoidGradient(z2)];
  % size(sg2)
  % sz_Theta2 = size(Theta2)
  % sz_delta3 = size(delta3)
  % sz_sg2 = size(sg2)
  % input("size all");
  delta2 = Theta2' * delta3 .* sg2;
  % delta2
  % input("delta 2")
  % size(delta2)
  %input("delta2");

  % dt2a1 = delta2(2:end) * a1'
  % input("dt2a1");
  Theta1_grad += delta2(2:end) * a1';
  Theta2_grad += delta3 * a2';
  
endfor

Theta1_grad /= m;
Theta2_grad /= m;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
