import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearReg import LinearRegression

# Creating a linear dataset to test the model 
X,y = datasets.make_regression(n_samples=500, n_features=1, noise=20, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=2)

# Creating two instances of the linear regression class model, to compare learning rates
# using the "fit" method defined in LinearReg.py to find optimal weights and bias
# default learning rate is 0.001
lin_regressor_1 = LinearRegression()
lin_regressor_1.fit(x_train,y_train)

lin_regressor_2 = LinearRegression(lr=0.05)
lin_regressor_2.fit(x_train, y_train)

# saving the predictions
prediction_1 = lin_regressor_1.predict(x_test)
prediction_2 = lin_regressor_2.predict(x_test)

# calling the mse method to return the mean squared error of the model on the data
# mse_1 = lin_regressor_2.mse(prediction_1, y_test)
# mse_2 = lin_regressor_2.mse(prediction_2, y_test)
mse_1 = np.sum(prediction_1 - y_test) ** 2
mse_2 = np.sum(prediction_2 - y_test) ** 2
print('MSE for model with learning rate {}, is {}'.format(lin_regressor_1.learningRate, mse_1))
print('MSE for model with learning rate {}, is {}'.format(lin_regressor_2.learningRate, mse_2))

# # calculating the predicted for plotting
# weights_1 = lin_regressor_1.weights
# bias_1 = lin_regressor_1.bias
# y_1 = (weights_1 * x_test) + bias_1

# weights_2 = lin_regressor_2.weights
# bias_2 = lin_regressor_2.bias
# y_2 = (weights_2 * x_test) + bias_2

'''
# matplot graphing
fig = plt.figure(figsize=(8,6))
plt.scatter(x_test[:,0], y_test, color='b', marker='o', s=30)
plt.plot(x_test, y_1, color='r')
plt.plot(x_test, y_2, color='g')
plt.legend(['data points','LR = 0.001', 'LR = 0.01'])
plt.show()
'''