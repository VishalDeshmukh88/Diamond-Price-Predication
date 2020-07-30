import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

data = pd.read_csv("diamonds.csv")
print(data.head())
label_encoder=preprocessing.LabelEncoder()
data['cut']=label_encoder.fit_transform(data['cut'])
data['color']=label_encoder.fit_transform(data['color'])

x = data.loc[:,['carat','cut','color','depth']].values
y = data.loc[:,'price'].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
errvalue = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = KNeighborsRegressor(n_neighbors = K)
    model.fit(xtrain, ytrain)
    yprd=model.predict(xtest)
    error = np.sqrt(mean_squared_error(ytest,yprd))
    errvalue.append(error)


wt = float(input("Enter the weight\n"))
ct=float(input("Enter the Cut( 0-4)"))
cl=float(input("Enter the color(0-6)"))
dt = float(input("Enter the Depth\n"))

xnew = [[wt,ct,cl,dt]]
ynew = model.predict(xnew)
print("Diamond Price", ynew[0])



'''plt.scatter(ytest,yprd)
plt.plot(ytest,yprd)
plt.title("Expected Price and Predict Output")
plt.xlabel("Excpected Value")
plt.ylabel("Predict output")
plt.show()
'''