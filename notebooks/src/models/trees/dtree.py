import numpy as np




class Node():
    """
    Implements an individual node in the Decision Tree. 
    """      
    
    def __init__(self, y):
        self.y = y                 # Labels of sample assigned to that node
        self.score = np.inf        # Score of the node (measure of impurity)
        self.feature_idx = None    # The feature used for the split (column number)
        self.threshold = None      # Threshold used of the splot (scalar value)
        self.left_child = None     # Left child of the node (of type Node)
        self.right_child = None    # Right child of the node (of type Node)
        
    def is_leaf(self):
        if self.feature_idx is None:
            return True
        else:
            return False
        
    def __str__(self):
        if self.is_leaf() == True:
            return f"Leaf, gini: {self.score:.3f}, samples: {self.y}"
        else:
            return f"X[{self.feature_idx}], threshold={self.threshold:.3g}, score={self.score:.3g}, samples={self.y}"




class SeleneDecisionTree:

    def __init__(self, max_depth: int=None, min_samples_split: int=2, max_features: int=None):        
        # Just a check if the parameter values are meaningful
        if max_depth is not None and max_depth < 1:
            raise Exception('If specified, max_depth must be greater or equal to 0')
        if min_samples_split is not None and min_samples_split < 1:
            raise Exception('If specified, min_samples_split must be greater or equal to 0')
        if max_features is not None and max_features < 1:
            raise Exception('If specified, max_features must be greater or equal to 1')
            
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def compute_thresholds(self, feature_values):
        # Get unique values to handle duplicates; return values will already be sorted
        values_sorted = np.unique(feature_values)
        #
        return (values_sorted[:-1] + values_sorted[1:]) / 2.0    

    def generate_split(self, feature_values: list[int|float], threshold: int|float):
        # Get all row indices where the value is <= threshold
        indices_left  = np.where(feature_values <= threshold)[0]
        # Get all row indices where the value is > threshold
        indices_right = np.where(feature_values > threshold)[0]
        # Return indices
        return indices_left, indices_right      

    def sample_feature_indices(self, X):
        n_features = X.shape[1]
        if self.max_features is not None:
            n_features = min(n_features, self.max_features)
        return np.random.choice(np.arange(X.shape[1]), size=n_features, replace=False)
    
    def find_best_feature_split(self, x, y, split_scoring_func):
        # Initialize the return values
        best_score, best_threshold, best_split = np.inf, None, None
        # Compute all possible thresholds
        thresholds = self.compute_thresholds(x)
        # Check all thresholds/partitions to find the one yielding the lowest score
        for threshold in thresholds:
            # Generate the split for the current threshold/partition
            split = self.generate_split(x, threshold)
            # Split the target values w.r.t. indices
            y_left, y_right = y[split[0]], y[split[1]]
            # Compute the score for the current split
            score = split_scoring_func(y_left, y_right)
            # Keep track of the key information of the split with the lowest score
            if score < best_score:
                best_score, best_threshold, best_split = score, threshold, split
        # Return key information of best split
        return best_score, best_threshold, best_split
    
    def find_best_split(self, X, y, split_scoring_func):
        # Initialize the return values
        best_score, best_threshold, best_fidx, best_split = np.inf, None, None, None
        # Perform feature sampling
        sampled_feature_indices = self.sample_feature_indices(X)
        # Check for each feature (i.e., each column in X), which split has the best (lowest) score
        for fidx in sampled_feature_indices:
            # Extract feature values from datasets
            x = X[:,fidx]
            # Calculate the best split for the current column/feature
            score, threshold, split = self.find_best_feature_split(x, y, split_scoring_func)
            # Keep track of the key information of the split with the lowest score
            if score <= best_score:
                best_score, best_split, best_fidx, best_threshold = score, split, fidx, threshold
        # Return the best split together with the relevant information
        return best_score, best_threshold, best_fidx, best_split
    
    def fit(self, X, y, node_scoring_func, split_scoring_func):
        # Initialize Decision Tree as a single root node
        self.tree = Node(y)
        # Start recursive building of Decision Tree
        self._fit(X, y, self.tree, node_scoring_func, split_scoring_func)
        # Return Decision Tree object
        return self

    def _fit(self, X, y, node, node_scoring_func, split_scoring_func, depth=0):
        # Calculate and set score of the node itself
        node.score = node_scoring_func(y)
        ### Check stop criteria ###############################################################
        # Stop splitting if we reach the max_depth
        if self.max_depth is not None and depth >= self.max_depth:
            return          
        # Stop splitting if the node has less then min_samples_split samples
        if self.min_samples_split is not None and self.min_samples_split > len(node.y):
            return        
        # If all targets are the same, no need for any further splitting
        if len(np.unique(y)) == 1:
            return
        #########################################################################################
        # Calculate the best split
        score, threshold, idx, split = self.find_best_split(X, y, split_scoring_func)
        # If the information gain is negative, no need for further splitting
        if score > node.score:
            return
        # Split the input and labels using the indices from the split
        X_left, X_right = X[split[0]], X[split[1]]
        y_left, y_right = y[split[0]], y[split[1]]
        # Update the parent node based on the best split
        node.feature_idx = idx
        node.threshold = threshold
        node.left_child = Node(y_left)
        node.right_child = Node(y_right)
        # Recursively fit both child nodes (left and right)
        self._fit(X_left, y_left, node.left_child, node_scoring_func, split_scoring_func, depth=depth+1)
        self._fit(X_right, y_right, node.right_child, node_scoring_func, split_scoring_func, depth=depth+1)   
    
    ###############################################################################################
    ### Prediction
    ###############################################################################################
        
    def get_targets(self, X):
        # Return list of individually predicted labels
        return [ self.get_targets_for_sample(self.tree, x) for x in X ]

    def get_targets_for_sample(self, node, x):
        # If the node is a leaf, simple return all targets
        if node.is_leaf():
            return node.y
        # If the node is not a leaf, go down the left or right subtree depending on threshold
        go_left = False
        if x[node.feature_idx] <= node.threshold:
            go_left = True
        if go_left:
            return self.get_targets_for_sample(node.left_child, x)
        else:
            return self.get_targets_for_sample(node.right_child, x)        


    ###############################################################################################
    ### Printing
    ###############################################################################################
    
    def __str__(self):
        self.print_tree(self.tree)
        return ''

    def print_tree(self, node, level=0):
        print('---'*level, node)
        if node.left_child is not None:
            self.print_tree(node.left_child, level=level+1)
        if node.right_child is not None:
            self.print_tree(node.right_child, level=level+1)



class SeleneDecisionTreeClassifier(SeleneDecisionTree):

    def __init__(self, max_depth: int=None, min_samples_split: int=2, max_features: int=None):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)

    
    def compute_gini_score_node(self, y):
        # Count the number of occurcences of output classes in split
        _, counts = np.unique(y, return_counts=True)
        # Calculate and return the Gini score
        return 1 - np.sum(np.square(counts/len(y)))
    
    
    def compute_gini_score_split(self, y_left, y_right):
        # Calculate the Gini score for the left and right node
        gini_score_left  = self.compute_gini_score_node(y_left)
        gini_score_right = self.compute_gini_score_node(y_right)
        # Calculate and return the weighted average Gini score
        return   len(y_left)/(len(y_left)+len(y_right))*gini_score_left \
               + len(y_right)/(len(y_left)+len(y_right))*gini_score_right

    def fit(self, X, y):
        return super().fit(X, y, self.compute_gini_score_node, self.compute_gini_score_split)

    def predict(self, X):
        # Get the targets for all samples
        targets = self.get_targets(X)
        # Return the majory class for each sample
        return [ np.bincount(t).argmax() for t in targets ]




class SeleneDecisionTreeRegressor(SeleneDecisionTree):

    def __init__(self, max_depth: int=None, min_samples_split: int=2, max_features: int=None):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)

    def compute_rss_score_node(self, y):
        # Compute the mean of both child nodes
        mean = np.mean(y) 
        # Calculate the RSS score
        rss_score = np.sum(np.square(y - mean)) / len(y)
        # Return the final RSS score
        return rss_score
    
    def compute_rss_score_split(self, y_left, y_right):
        # Compute the mean of both child nodes
        mean_left  = np.mean(y_left) 
        mean_right = np.mean(y_right)
        # Calculate the RSS score for the left and right node
        rss_score_left  = np.sum(np.square(y_left - mean_left)) / len(y_left)
        rss_score_right = np.sum(np.square(y_right - mean_right)) / len(y_right)
        # Calculate and return the weighted average Gini score
        return   len(y_left)/(len(y_left)+len(y_right))*rss_score_left \
               + len(y_right)/(len(y_left)+len(y_right))*rss_score_right

    def fit(self, X, y):
        return super().fit(X, y, self.compute_rss_score_node, self.compute_rss_score_split)

    def predict(self, X):
        # Get the targets for all samples
        targets = self.get_targets(X)
        # Return the majory class for each sample
        return [ np.mean(t) for t in targets ]
