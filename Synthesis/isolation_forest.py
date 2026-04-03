import numpy as np
import pandas as pd
import random
import math

class iTree:
    """A single Isolation Tree"""
    def __init__(self, X, current_height, limit):
        self.height = current_height
        self.size = len(X)
        
        # Stop growing the tree if limit is reached or node has 1 or 0 items
        if current_height >= limit or self.size <= 1:
            self.node_type = 'exNode' # External node (leaf)
            self.split_col = None
            self.split_val = None
            self.left = None
            self.right = None
        else:
            # Randomly select a feature
            self.split_col = random.randint(0, X.shape[1] - 1)
            min_val = X[:, self.split_col].min()
            max_val = X[:, self.split_col].max()
            
            # If all values are identical, make it a leaf
            if min_val == max_val:
                self.node_type = 'exNode'
                self.split_col = None
                self.split_val = None
                self.left = None
                self.right = None
            else:
                # Randomly select a split point between min and max
                self.split_val = random.uniform(min_val, max_val)
                
                # Split the data
                left_idx = X[:, self.split_col] < self.split_val
                right_idx = ~left_idx
                
                self.node_type = 'inNode' # Internal node
                self.left = iTree(X[left_idx], current_height + 1, limit)
                self.right = iTree(X[right_idx], current_height + 1, limit)

class IsolationForest:
    """From-scratch implementation of the Isolation Forest algorithm"""
    def __init__(self, n_estimators=100, contamination=0.1, max_samples='auto', random_state=None):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.trees = []
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def _c_factor(self, n):
        """Adjustment factor for unbuilt tree depth"""
        if n > 2:
            return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) * 1.0 / n)
        elif n == 2:
            return 1.0
        else:
            return 0.0

    def fit_predict(self, X):
        # Convert to numpy array if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        n_samples = X.shape[0]
        
        if self.max_samples == 'auto':
            self.max_samples = min(256, n_samples)
            
        # Height limit for trees (l = ceiling(log2(subsample_size)))
        limit = int(math.ceil(math.log2(self.max_samples)))
        
        # Build the forest
        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, self.max_samples, replace=False)
            X_sub = X[idx]
            self.trees.append(iTree(X_sub, 0, limit))
            
        # Calculate expected path length for each observation
        path_lengths = np.zeros(n_samples)
        for i in range(n_samples):
            paths = [self._path_length(X[i], tree, 0) for tree in self.trees]
            path_lengths[i] = np.mean(paths)
            
        # Calculate anomaly scores
        c = self._c_factor(self.max_samples)
        scores = 2.0 ** (-path_lengths / c)
        
        # Determine the threshold based on the requested contamination rate
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        # Return -1 for Anomaly, 1 for Normal
        return np.where(scores > threshold, -1, 1)

    def _path_length(self, x, tree, current_height):
        if tree.node_type == 'exNode':
            return current_height + self._c_factor(tree.size)
        
        if x[tree.split_col] < tree.split_val:
            return self._path_length(x, tree.left, current_height + 1)
        else:
            return self._path_length(x, tree.right, current_height + 1)