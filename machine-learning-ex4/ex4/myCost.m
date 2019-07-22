function J = myCost(y, x, theta1, theta2)

f = @(tx) ((-y * log(tx)) - ((1 - y) * log(1 - tx)));

[a1, z2, a2, z3, a3] = myCostStep(y, x, theta1, theta2);

J = f(a3);

end
