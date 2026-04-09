import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        # check if n_feats is set, otherwise use all of them
        if not self.n_feats:
            self.n_feats = X.shape[1]
        else:
            self.n_feats = min(self.n_feats, X.shape[1])
            
        self.root = self._make_tree(X, y, 0)

    def _make_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stop if we hit max depth, or all labels are the same, or not enough samples
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_val = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_val)

        # pick random features for this split
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        best_feat, best_thresh = self._find_best_split(X, y, feat_idxs)
        
        # if we couldn't find a good split, make  a leaf
        if best_feat is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=Counter(y).most_common(1)[0][0])

        left_child = self._make_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._make_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left_child, right_child)

    def _find_best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx = None
        split_thresh = None
        
        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            
           # check 25 different thresholds
            thresholds = np.percentile(X_col, np.linspace(4, 96, 25))
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                gain = self._info_gain(y, X_col, thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _info_gain(self, y, X_col, thresh):
        # calculate parent gini impurity
        counts = np.bincount(y, minlength=3)
        probs = counts / len(y)
        parent_gini = 1.0 - np.sum(probs ** 2)
        
        left_idxs = np.argwhere(X_col <= thresh).flatten()
        right_idxs = np.argwhere(X_col > thresh).flatten()
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate child gini
        n = len(y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)
        
        # left side gini
        counts_l = np.bincount(y[left_idxs], minlength=3)
        probs_l = counts_l / len(left_idxs)
        e_l = 1.0 - np.sum(probs_l ** 2)
        
        # right side gini
        counts_r = np.bincount(y[right_idxs], minlength=3)
        probs_r = counts_r / len(right_idxs)
        e_r = 1.0 - np.sum(probs_r ** 2)
        
        # weighted average of the children
        child_gini = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_gini - child_gini

    def predict(self, X):
        results = []
        for x in X:
            results.append(self._traverse(x, self.root))
        return np.array(results)

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            print(f"training tree {i+1} out of {self.n_trees}...")
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_feats=self.n_features
            )
            
            # sampling with replacement
            n_samples = X.shape[0]
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict_proba(self, X):
        tree_preds = []
        for tree in self.trees:
            tree_preds.append(tree.predict(X))
            
        tree_preds = np.array(tree_preds) 
        tree_preds = np.swapaxes(tree_preds, 0, 1) 
        
        # calculate the probs for each class
        probas = []
        for pred in tree_preds:
            counts = Counter(pred)
            p0 = counts.get(0, 0) / self.n_trees
            p1 = counts.get(1, 0) / self.n_trees
            p2 = counts.get(2, 0) / self.n_trees
            probas.append([p0, p1, p2])
            
        return np.array(probas)