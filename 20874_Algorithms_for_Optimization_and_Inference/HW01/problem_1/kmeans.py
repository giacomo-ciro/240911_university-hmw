import numpy as np

class KMeans:
    def __init__(self, n_clusters=None, init='random', n_init=10, tol = 1e-4, normalize=False):

        if n_clusters is None:
            raise ValueError('Number of clusters must be specified.')
        
        self.n_clusters = n_clusters
        self.n_init = 1 if init == 'frequency' else n_init
        self.init = init
        self.tol = tol
        self.normalize = normalize

    def fit(self, X, max_iters=300):
        """
        Fit the model to the data X.
        X is an array of shape (n_samples, n_features).
        Args:
            X: np.array
                Data to fit the model on.
            max_iters: int
                Maximum number of iterations for one run of Lloyd's algorithm.

        """
        
        if self.normalize:
            self.std = X.std(axis=0)
            self.mean = X.mean(axis=0)
            X = (X - self.mean) / self.std

        best_obj = np.inf

        for _ in range(self.n_init):
            
            centroids, new_obj = self.lloyd(X, max_iters=max_iters)
            
            print(f'Iteration: {_+1:03} | Objective: {new_obj:,.2f}')
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_centroids = centroids
                print('New best objective:', best_obj)
        
        if self.normalize:
            best_centroids = best_centroids * self.std + self.mean
            X = X * self.std + self.mean

        self.centroids = best_centroids
        self.obj = best_obj
        self.labels = self.predict(X)
        
        return f'Fitting done.\nCentroids stored in self.\nObjective value: {self.obj}.'
            
    def lloyd(self, X, max_iters):

        """"
        One complete run of Lloyd's algorithm:
            1. Init centroids
            2. Assign each sample to the closest centroid
            3. Update the centroids 
        Returns the centroids and the objective value.
        
        """
        if self.init == 'random':
            centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        elif self.init == 'frequency':
            points, counts = np.unique(X, axis=0, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            centroids = points[sorted_indices[:self.n_clusters]]
        
        elif self.init == 'frequency+':
            points, counts = np.unique(X, axis=0, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            centroids = points[sorted_indices[:self.n_clusters]]
            centroids += np.random.normal(0, 1, centroids.shape).astype(np.uint8)
        
        iter = 0
        while iter < max_iters:

            distances = np.linalg.norm(X[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(self.n_clusters)])
            
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                centroids = new_centroids
                print(f'Converged in {iter} iterations.')
                break
            
            centroids = new_centroids
            iter += 1

        return centroids, np.min(distances, axis=1).sum()
        
    
    def predict(self, X):
        """
        Return the closest centroid for each sample in X.
        X is an array of shape (n_samples, n_features).

        """
        if getattr(self, 'centroids', None) is None:
            raise ValueError('Model not fitted.')
        return np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)