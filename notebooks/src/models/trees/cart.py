import numpy as np
from itertools import chain, combinations



class Node():
    """
    Implements an individual node in the Decision Tree. 
    """      
    
    def __init__(self, y):
        self.y = y                 # Labels of sample assigned to that node
        self.score = np.inf        # Gini score of the node (measure of impurity)
        self.feature_col = None    # The feature used for the split (column number)
        self.criterion = None      # Threshold used of the splot (scalar value)
        self.left_child = None     # Left child of the node (of type Node)
        self.right_child = None    # Right child of the node (of type Node)
        
    def is_leaf(self):
        if self.feature_col is None:
            return True
        else:
            return False
        
    def __str__(self):
        if self.is_leaf() == True:
            return f"Leaf, gini: {self.score:.3f}, samples: {self.y}"
        else:
            #return 'X[{}] <= {:.3f}, gini: {:.3f}, #samples: {}'.format(self.feature_col, self.criterion, self.score, len(self.y))
            return f"X[{self.feature_col}], criterion: {self.criterion}, gini: {self.score:.3f}, samples: {self.y}"



class SeleneCART:
    
    def __init__(self, feature_types, max_depth=None, min_samples_split=2):
        
        ## Just a check if the parameter values are meaningful
        if max_depth is not None and max_depth < 1:
            raise Exception('If specified, max_depth must be greater or equal to 0')
        if min_samples_split is not None and min_samples_split < 1:
            raise Exception('If specified, min_samples_split must be greater or equal to 0')
            
        self.tree = None
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    
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
    
    
    def compute_partitions(self, feature_values):
        partitions = []
        # Compute the set of all unique feature values
        values = set(feature_values)
        # Compute all subsets of features values (ignoring empty set and the full set)
        subsets = chain.from_iterable(combinations(values, r) for r in range(1, len(values)))
        # Initialize set of subset we have already seen
        seen_subsets = set()
        # Loop over all subsets and consider them as the "left branch condition"
        for left in subsets:
            # Convert subset from tuple to true set
            left = set(left)
            # Compute "right branch condition" using the set difference
            right = values - left
            # If we have already seen the right subset, we can stop
            if frozenset(right) in seen_subsets:
                break
            # Add current left subset to set of seen subsets
            seen_subsets.add(frozenset(left))
            # Add current split to final result list
            partitions.append((left, right))
        # Return all possible splits
        return partitions
    

    def compute_thresholds(self, feature_values):
        # Get unique values to handle duplicates; return values will already be sorted
        values_sorted = np.unique(feature_values)
        #
        return (values_sorted[:-1] + values_sorted[1:]) / 2.0    


    def generate_split_nominal(self, feature_values, partition):
        indices_left  = [ idx  for idx, val in enumerate(feature_values) if val in partition[0] ]
        indices_right = [ idx  for idx, val in enumerate(feature_values) if val in partition[1] ]
        return indices_left, indices_right    

    def generate_split_general(self, feature_values, threshold):
        # Get all row indices where the value is <= threshold
        indices_left  = np.where(feature_values <= threshold)[0]
        # Get all row indices where the value is > threshold
        indices_right = np.where(feature_values > threshold)[0]
        # Return indices
        return indices_left, indices_right      
    
    def find_best_feature_split(self, x, y, feature_type):
        ## Initialize the return values
        best_score, best_criterion, best_split = np.inf, None, None
    
        # Create splits depending on the feature being nominal or nonnominal
        if feature_type == "nominal":
            criterions = self.compute_partitions(x)
            generate_split = self.generate_split_nominal
        else:
            criterions = self.compute_thresholds(x)
            generate_split = self.generate_split_general
    
        # Check all thresholds/partitions to find the one yielding the lowest Gini score
        for criterion in criterions:
            # Generate the split for the current threshold/partition
            split = generate_split(x, criterion)
            # Split the target values w.r.t. indices
            y_left, y_right = y[split[0]], y[split[1]]
            # Compute the Gini score for the current split
            score = self.compute_gini_score_split(y_left, y_right)
            # Keep track of the key information of the split with the lowest Gini score
            if score < best_score:
                best_score, best_criterion, best_split = score, criterion, split
    
        # Return key information of best split
        return best_score, best_criterion, best_split
    
    
    def find_best_split(self, X, y):
        # Initialize the return values
        best_score, best_criterion, best_col, best_split = np.inf, None, None, None
        # Check for each feature (i.e., each column in X), which split has the best (lowest) score
        for col in range(X.shape[1]):
            # Extract feature values from datasets
            x = X[:,col]
            # Calculate the best split for the current column/feature
            score, threshold, split = self.find_best_feature_split(x, y, self.feature_types[col])
            # Keep track of the key information of the split with the lowest Gini score
            if score <= best_score:
                best_score, best_split, best_col, best_threshold = score, split, col, threshold
        # Return the best split together with the relevant information
        return best_score, best_threshold, best_col, best_split
    
    
    
    def fit(self, X, y):
        # Initializa Decision Tree as a single root node
        self.tree = Node(y)
        # Start recursive building of Decision Tree
        self._fit(X, y, self.tree)
        # Return Decision Tree object
        return self


    def _fit(self, X, y, node, depth=0):
        # Calculate and set Gini score of the node itself
        node.score = self.compute_gini_score_node(y) 
        ### Check stop criteria ###############################################################
        # Stop splitting if we reach the max_depth
        if self.max_depth is not None and depth >= self.max_depth:
            return    
        # Stop splitting if the node has less then min_samples_split samples
        if self.min_samples_split is not None and self.min_samples_split >= len(node.y):
            return        
        # If all class labels are the same, no need for any further splitting
        if len(np.unique(y)) == 1:
            return
        #########################################################################################
        # Calculate the best split
        score, criterion, col, split = self.find_best_split(X, y)
        # If the information gain is negative, no need for further splitting
        if score > node.score:
            return
        # Split the input and labels using the indices from the split
        X_left, X_right = X[split[0]], X[split[1]]
        y_left, y_right = y[split[0]], y[split[1]]
        # Update the parent node based on the best split
        node.feature_col = col
        node.criterion = criterion
        node.left_child = Node(y_left)
        node.right_child = Node(y_right)
        # Recursively fit both child nodes (left and right)
        self._fit(X_left, y_left, node.left_child, depth=depth+1)
        self._fit(X_right, y_right, node.right_child, depth=depth+1)   



    
    ###############################################################################################
    ### Prediction
    ###############################################################################################
        
    def predict(self, X):
        # Return list of individually predicted labels
        return np.array([ self.predict_sample(self.tree, x) for x in X ])


    def predict_sample(self, node, x):
        # If the node is a leaf, return the class with highest probability
        # (this can happen of in the leaf are still different classes)
        if node.is_leaf():
            #
            vals, counts = np.unique(node.y, return_counts=True)
            #
            return vals[np.argmax(counts)]
        # If the node is not a leaf, go down the left or right subtree depending on criterion
        go_left = False
        if self.feature_types[node.feature_col] == "nominal":
            if x[node.feature_col] in node.criterion[0]:
                go_left = True
        else:
            if x[node.feature_col] <= node.criterion:
                go_left = True
        if go_left:
            return self.predict_sample(node.left_child, x)
        else:
            return self.predict_sample(node.right_child, x)        


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
        