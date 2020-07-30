import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


data = pd.read_csv("diamonds.csv")
print(data.head())

label_encoder=preprocessing.LabelEncoder()
data['cut']=label_encoder.fit_transform(data['cut'])
data['color']=label_encoder.fit_transform(data['color'])
print(data['cut'].unique())
print(data['color'].unique())


x = data.loc[:,['carat','cut','color','depth']].values
y = data.loc[:,'price'].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
regressor=DecisionTreeRegressor()
regressor.fit(xtrain, ytrain)
yprd = regressor.predict(xtest)

wt = float(input("Enter the weight\n"))
ct=float(input("Enter the Cut( 0-4)"))
cl=float(input("Enter the color(0-6)"))
dt = float(input("Enter the Depth\n"))

xnew = [[wt,ct,cl,dt]]
ynew = regressor.predict(xnew)
print("Diamond Price", ynew[0])

print("Mean Absolute Error", metrics.mean_absolute_error(ytest, yprd))
print("Mean Squared Error" ,metrics.mean_squared_error(ytest, yprd))
print("Root mean Squared Error ", np.sqrt(metrics.mean_squared_error(ytest, yprd)))
print("Accuray",accuracy_score(ytest,yprd)*100)
