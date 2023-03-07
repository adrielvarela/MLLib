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

    def build_tree(self, dataset, curr_depth):
        X,Y = dataset[:,:-1], dataset[:,-1]
