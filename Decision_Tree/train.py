import numpy as np
import pandas as pd
from sklearn.model_selection import test_train_split

data = pd.read_csv("winequality-red.csv")
print(data.head())

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values