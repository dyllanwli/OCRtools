from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets




reg = linear_model.Ridge(alpha = 0.5)

reg.fit(bostron.data)


lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target



predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()