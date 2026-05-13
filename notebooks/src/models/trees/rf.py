import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor




class SeleneRandomForest():

    def __init__(self, n_estimators: int=100, max_depth: int=None, min_samples_split: int=2, max_features: int=None):
        # Check if all arguments have meaningful values
        assert n_estimators is not None and n_estimators > 0, "The number of estimators must be greater than 0"
        assert max_depth is None or max_depth > 0, "The maximum depth must be greater than 0"
        assert min_samples_split is not None and min_samples_split > 1, "The minimum numbers of samples in a node must be greater than 1"
        assert max_features is None or max_features > 0, "The maximum number of features must be greater than 0"
        #
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def create_bootstrap_sample(self, X, y):
        X_bootstrap, y_bootstrap = None, None
        #
        N, d = X.shape
        #
        random_sample_indices = np.random.choice(N, N, replace=True)
        #
        X_bootstrap = X[random_sample_indices]
        y_bootstrap = y[random_sample_indices]
        #
        return X_bootstrap, y_bootstrap

    def sample_feature_indices(X, max_features: int=None):
        # Get the number of available features
        n_features = X.shape[1]
        # Compute sample size as the minimum of available features and max_features value
        if max_features is not None and max_features > 1:
            n_features = min(n_features, max_features)
        # Return a random sample of available indices of size n_features (without replacement)
        return np.random.choice(np.arange(X.shape[1]), size=n_features, replace=False)

    def fit(self, X, y, DecisionTreeClass):
        self.trees = []
    
        for _ in range(self.n_estimators):
            X_bootstrap, y_bootstrap = self.create_bootstrap_sample(X, y)
            tree = DecisionTreeClass(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            tree = tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
        return self


class SeleneRandomForestClassifier(SeleneRandomForest):

    def __init__(self, n_estimators: int=100, max_depth: int=None, min_samples_split: int=2, max_features: int=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)

    def fit(self, X, y):
        return super().fit(X, y, DecisionTreeClassifier)

    def predict(self, X):
        # Pass X to all trees to get all predictions
        ys = np.asarray([ tree.predict(X) for tree in self.trees ])
        # Compute the majority class label for each sample across all trees
        return np.array([ np.bincount(ys[:, col]).argmax() for col in range(ys.shape[1]) ])


class SeleneRandomForestRegressor(SeleneRandomForest):

    def __init__(self, n_estimators: int=100, max_depth: int=None, min_samples_split: int=2, max_features: int=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)

    def fit(self, X, y):
        return super().fit(X, y, DecisionTreeRegressor)

    def predict(self, X):
        # Pass X to all trees to get all predictions
        ys = np.asarray([ tree.predict(X) for tree in self.trees ])
        # Compute the mean for each sample across all trees
        return np.mean(ys, axis=0)