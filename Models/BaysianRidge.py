import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import BayesianRidge
from sklearn import preprocessing

data = pd.read_csv("diamonds.csv")
print(data.head())
label_encoder=preprocessing.LabelEncoder()
data['cut']=label_encoder.fit_transform(data['cut'])
data['color']=label_encoder.fit_transform(data['color'])

x = data.loc[:,['carat','cut','color','depth']].values
y = data.loc[:,'price'].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
model=BayesianRidge(compute_score=True)
model.fit(xtrain,ytrain)

yprd=model.predict(xtest)


wt = float(input("Enter the weight\n"))
ct=float(input("Enter the Cut( 0-4)"))
cl=float(input("Enter the color(0-6)"))
dt = float(input("Enter the Depth\n"))

xnew = [[wt,ct,cl,dt]]
ynew = model.predict(xnew)
print("Diamond Price", ynew[0])

print("Mean sqaured Error",mean_squared_error(ytest,yprd))
print("Variance Score",r2_score(ytest,yprd))
print("Coefficent",model.coef_)
print("Intercept ",model.intercept_)
print("Accuracy",model.score(x,y)*100)

'''plt.scatter(ytest,yprd)
plt.plot(ytest,yprd)
plt.title("Expected Price and Predict Output")
plt.xlabel("Excpected Value")
plt.ylabel("Predict output")
plt.show()
'''