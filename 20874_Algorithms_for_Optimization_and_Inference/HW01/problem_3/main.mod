
set I;  # total colors
set J;  # available centroids

param d{i in I, j in J};    # distance between color i and centroid j
param w{i in I};            # weight of color i
param k;                    # number of centroids to use
param N;                    # number of colors

var x{i in I, j in J} binary;  # 1 if color i is assigned to centroid j
var y{j in J} binary;          # 1 if centroid j is used

minimize obj:
    sum{i in I, j in J} w[i] * d[i, j] * x[i, j];

subject to c1{j in J}: sum{i in I} x[i, j] <= y[j] * N;       # y[j] = 1 if centroid j is used
subject to c2: sum{j in J} y[j] = k;                        # exactly k centroids are used
subject to c3{i in I}: sum{j in J} x[i, j] = 1;             # each color is assigned to exactly one centroid

solve;

display obj, {i in I, j in J: x[i, j] == 1};

end;