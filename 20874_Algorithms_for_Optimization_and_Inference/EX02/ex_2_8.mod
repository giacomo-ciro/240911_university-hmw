var x1 >= 0;
var x2 >= 0;

maximize obj: x1 - x2;
s.t. con1: x1 - x2 <=0.5;
s.t. con2: x1 >=1;

solve;

display x1, x2, obj;

end;
