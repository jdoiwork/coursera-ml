function v = labelAsVector(label, num)

temp = zeros(1, num);

temp(label) = 1;

v = temp;

end
