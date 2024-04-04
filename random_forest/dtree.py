import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)
        
    
    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node. This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if isinstance(self, LeafNode):
            return self
        
        if x_test[self.col] < self.split: # check left
            if isinstance(self.lchild, LeafNode): # if it is leaf node return node itself
                return self.lchild
            else:
                return self.lchild.leaf(x_test)
        else: # Check right, the same process
            if isinstance(self.rchild, LeafNode):
                return self.rchild
            else:
                return self.rchild.leaf(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        # self.y = y 

    def predict(self, x_test):
        return self.prediction


def gini(x):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    counts = np.unique(x,return_counts=True)[1]
    gini_score = 1 - np.sum((counts/sum(counts))**2)
    
    return gini_score


## Modify!
def find_best_split(X, y, loss, min_samples_leaf, max_features):
    k = 11
    best = [-1,-1,loss(y)] #feature, split, loss function
    n,p = X.shape
    feature_count = int(max_features*p)
    features = np.random.choice(range(p), size=feature_count, replace=False)
    for i in features:
        k = 11
        column_data = X[:,i]
        # not sure if random k should be with or without replacement
        if len(column_data) < 11:
            k = len(column_data)
       
        sample_candidates = np.random.choice(column_data, size=k,replace=False) 

        for split in sample_candidates:

            yl = y[column_data < split]
            yr = y[column_data >= split]

            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue
                
            l = (len(yl)*loss(yl) + len(yr)*loss(yr)) / (len(yl)+len(yr))
            
            if l == 0:
                
                return best[0],best[1]
            
            if l < best[2]:
                
                best = [i,split,l]
                
    return best[0],best[1]
 

class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None, max_features = 0.3):
        
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        self.max_features = max_features
        
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)



    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf:
            return self.create_leaf(y)
        col, split = find_best_split(X,y,self.loss,self.min_samples_leaf, self.max_features)
        
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X.T[col] < split],y[X.T[col] < split])
        rchild =  self.fit_(X[X.T[col] >= split],y[X.T[col] >= split])
        
        return DecisionNode(col,split,lchild,rchild)
        
        

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        y_results = []
        for x_test in X_test:
            result = self.root.predict(x_test)
            y_results.append(result)

        return np.array(y_results)

         

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features = 0.3):
        super().__init__(min_samples_leaf, loss=np.var, max_features = max_features)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        predicted_y = self.predict(X_test)
        return r2_score(predicted_y,y_test)



    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y,np.mean(y))



class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features = 0.3):
        super().__init__(min_samples_leaf, loss=gini, max_features = max_features)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        prediction = self.predict(X_test)
        
        return  accuracy_score(prediction,y_test)

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y,stats.mode(y, keepdims=True)[0])
