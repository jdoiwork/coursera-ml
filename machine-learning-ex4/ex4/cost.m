function J = cost(y, x, theta1, theta2)

% size(x) % => 1, 400
% input("size of x");

% size(Theta1) % => 25, 401
% input("size of theta1"); 

% size(Theta2) % => 10, 26
% input("size of theta2"); 

% hypo = @(t, x) sigmoid(t * x);
f = @(tx) ((-y * log(tx)) - ((1 - y) * log(1 - tx)));

% a1 = [1; x];

% % theta1 * a1
% % input("theta1 * a1");

% a2 = hypo(theta1,  a1);
% a2 = [1; a2];

% % z2
% % input("z2")
% % a2
% % input("a2")

% a3 = hypo(theta2, a2);

[a1, z2, a2, z3, a3] = costStep(y, x, theta1, theta2);

J = f(a3);

end
