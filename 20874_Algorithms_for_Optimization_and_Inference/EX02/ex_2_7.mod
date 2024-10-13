var x1 >= 0;
var x2 >= 0;

maximize obj: 10*x1 + 6*x2;
s.t. con1: 5*x1 - x2 <= 12;
s.t. con2: 2*x1 + 3*x2 <= 6;
s.t. branch1: x1<=2;
s.t. branch3: x2>=1; 
s.t. branch4: x1>=2;
solve;

display x1,x2, obj;

end;
