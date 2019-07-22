function J = myCost(tx, y)

J = (-y * log(tx)) - ((1 - y) * log(1 - tx));

end
