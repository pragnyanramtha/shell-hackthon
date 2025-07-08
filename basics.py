from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
import numpy as np
from sklearn.datasets import load_diabetes

def main():
    X,y = load_diabetes(return_X_y=True)
    print(X[0:5])
    mod = KNeighborsRegressor()
    mod.fit(X, y)
    print(mod.predict(X[0:5]))
    print( y[0:5])
    # reg = linear_model.LinearRegression()
    #reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])


main()
