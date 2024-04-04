import numpy as np
from sklearn.utils import resample
from sklearn.metrics import r2_score, accuracy_score

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False, max_features=0.3):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.max_features = max_features

    def calculate_oob_score(self, X, y):
        oob_predictions = []

        for i in range(len(X)):
            # Get all trees that hasn't trained on this obs
            oob_trees = []
            for j in range(self.n_estimators):
                if i in self.oob_indices_list[j]: 
                    oob_trees.append(self.trees[j])

            if oob_trees == []: 
            # When obs is used in all trees，oob_tree is empty
                if isinstance(self, RandomForestRegressor621):
                    oob_pred = np.mean(y) # default value
                else:
                    oob_pred = np.argmax(np.bincount(y)) # default value

            else:
                if isinstance(self, RandomForestRegressor621):
                    # Take average for regression
                    preds = []
                    for tree in oob_trees:
                        preds.append(tree.predict(X[i].reshape(1, -1)))
                    oob_pred = np.mean(preds)

                else:
                    # Classification: put topgether and count type vote
                    preds = []
                    for tree in oob_trees:
                        preds.append(tree.predict(X[i].reshape(1, -1)).astype(int))
                    preds = np.array(preds)

                    all_votes = preds.flatten() # to 1-D 
                    all_counts = np.bincount(all_votes)
                    oob_pred = np.argmax(all_counts)

            oob_predictions.append(oob_pred)

        # Scores
        if isinstance(self, RandomForestRegressor621):
            return r2_score(y, oob_predictions)
        else:
            return accuracy_score(y, oob_predictions)


    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.trees = []
        self.oob_indices_list = []  # List to store OOB indices for each tree

        for i in range(self.n_estimators):

            bootstrap_index = np.random.choice(range(len(X)), size=len(X), replace=True)

            oob_indices = []
            for i in range(len(X)):  # Iterate over all indices in the dataset
                if i not in bootstrap_index:  # Check if the index is not in the bootstrap sample
                    oob_indices.append(i) 
            self.oob_indices_list.append(oob_indices)

            if isinstance(self, RandomForestRegressor621): # Regression
                tree = RegressionTree621(min_samples_leaf=self.min_samples_leaf, max_features = self.max_features)
            else:    # Classification
                self.n_classes = len(np.unique(y))
                tree = ClassifierTree621(min_samples_leaf=self.min_samples_leaf, max_features = self.max_features)

            X_bootstrap = X[bootstrap_index]
            y_bootstrap = y[bootstrap_index]
            # Fit the tree
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

        if self.oob_score:
            self.oob_score_ = self.calculate_oob_score(X, y)



class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score, max_features = max_features)
        # self.trees = ... what we need this for?
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features


    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        weighted_sum = np.zeros(len(X_test))
        weights = np.zeros(len(X_test))

        for tree in self.trees:
            for i in range(len(X_test)):
                leaf = tree.root.leaf(X_test[i])
                weighted_sum[i] += leaf.prediction * leaf.n
                weights[i] += leaf.n
    
        weighted_average = weighted_sum / weights
        return weighted_average


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        predictions = self.predict(X_test)
        return r2_score(y_test, predictions)
        


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score, max_features = max_features)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        # self.trees = ... ??
        self.max_features = max_features


    def predict(self, X_test) -> np.ndarray:
        class_counts = np.zeros((len(X_test), self.n_classes))

        for tree in self.trees:
            for i, x in enumerate(X_test):
                leaf = tree.root.leaf(x)
                class_counts[i, leaf.prediction] += leaf.n 
                # All votes in node go to whatever leaf node predicts

        #Determine the class with the maximum count
        predictions = np.argmax(class_counts, axis=1)
        return predictions


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)