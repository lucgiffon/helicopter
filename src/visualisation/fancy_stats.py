# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:57:12 2019

@author: Jane
"""



    
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from yellowbrick.features import rank2d 
from yellowbrick.features import Manifold  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge 
from yellowbrick.regressor import ResidualsPlot

   
df = pd.read_csv('pcm_mixed.csv', encoding = 'ISO-8859-1', sep = ';')
df.replace('Infinity', np.nan, inplace = True)
df.replace(np.nan, 0, inplace = True)
print(df.columns)


X = df.iloc[:, 0:55].astype(float)
y = df.iloc[:, 55].astype(float)


for col in X.columns:
    print(col)
    sns.distplot(X[col])
    plt.savefig(fname = 'dist' + col + '.png', format = 'png')
    plt.show()
    break
   
visualizer = rank2d(X)
visualizer.show()

#-------------------------------------------
viz = Manifold(manifold = "isomap", n_neighbors = 20, target_type = "continuous")

viz.fit_transform(X, y)  # Fit the data to the visualizer
viz.show()               # Finalize and render the figure


# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Instantiate the linear model and visualizer
model = Ridge()
visualizer = ResidualsPlot(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure


from sklearn.gaussian_process import GaussianProcessRegressor

#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

kernel = RBF() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel)
print('score: ', gpr.score(X, y))

#print(gpr.predict(X[0, :], return_std = True))

model = GaussianProcessRegressor(kernel=kernel)#.fit(X_train, y_train)
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure


