set I;  # index set
param a {I};  # parameter
var x {I} >= 0;  # decision variable

maximize obj: sum{ i in I } a[i] * x[i];

s.t. c1: sum{ i in I } x[i] <= 10;