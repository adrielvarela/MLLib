import numpy as np
import pandas as pd


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None) -> None:
        # Decision Node
        # conditions defined by feature index and threshold
        self.feature_index = feature_index
        self.threshold = threshold

        # traversal
        self.left = left
        self.right = right

        # stores information gained by the spplit at node
        self.info_gain = info_gain

        # Leaf Nodes
        self.value = value

class DecisionTree():
    def __init__(self, min_sample_splits=2, max_depth=2) -> None:
        # initializing the root of the tree
        self.root = None

        # stopping conditions
        # min samples tells us when to stop, if node contains less than mininum number of samples it will stop
        self.min_sample_split = min_sample_splits
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X,Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

    

        if num_samples >= self.min_sample_splits and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)

            if best_split['info_gain'] > 0:

                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth+1)
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth+1)

                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree,
                best_split['info_gain'])

            leaf_value = self.calculate_leaf_value(Y)
            return Node(value=leaf_value)


    def get_best_split(self, dataset, num_samples, num_features):

        best_split = {}
        max_info_gain = float('-inf')

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]  
            possible_thresholds = np.unique(feature_values)

            for thresholds in possible_thresholds:

                dataset_left, dataset_right = self.split(dataset, feature_index, thresholds)

                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]

                    curr_info_gain = self.information_gain(y, left_y, right_y, 'gini')

                    if curr_info_gain>max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = thresholds
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain

        return (best_split)


    def split(self, dataset, feature_index, threshold):
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])

        return (dataset_left, dataset_right)
    
    def information_gain(self, parent, l_child, r_child, mode='entropy'):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if mode=='gini':
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        ent = 0
        for cls in class_labels:
            p_cls = len([y==cls]) / len(y)
            ent += -p_cls * np.log2(cls)
        return (ent)
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            gini += p_cls**2
        
        return (1 - gini)
    
    def calculate_leaf_val(self, Y):

        Y = list(Y)
        return(max(Y, key=Y.count))
    
    def print_tree(self, tree, indent='   '):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
        
    def fit(self, X, Y):
        dataset = np.concatenate((X,Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        predictions = [self.make_prediction(X, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
