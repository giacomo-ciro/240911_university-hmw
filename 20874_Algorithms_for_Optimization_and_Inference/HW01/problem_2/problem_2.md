## Problem 2: IP Modelling of K-means Clustering
Given $N$ total points, denote with $S_i$ all possible subsets of points for $i=1, ..., 2^N-1$ (we omit the empty cluster). Let $y_i \in \{0,1\}$ indicate whether the mean point of cluster $S_i$, denoted by $s_i$, is chosen as centroid.  
The k-means algorithm can be rewritten via IP as follows:

$$
\min_{y_i \in \{0,1\}} \sum_{i=1}^{2^N-1}y_i\sum_{x \in S_i}||s_i - x||^2\\
s.t. \sum_{i=1}^{2^N-1}y_i = k
$$

In words, we are chosing $k$ clusters among all possible ones such that the total distance of the points within each cluster and the respective cluster centroid is minimized, subject to choosing exactly $k$ clusters.