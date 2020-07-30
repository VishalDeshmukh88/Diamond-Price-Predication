import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

data = pd.read_csv("diamonds.csv")
print(data.head())
label_encoder = preprocessing.LabelEncoder()
data['cut'] = label_encoder.fit_transform(data['cut'])
data['color'] = label_encoder.fit_transform(data['color'])

x = data.loc[:, ['carat', 'cut', 'color', 'depth']].values
y = data.loc[:, 'price'].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(xtrain, ytrain)

yprd = model.predict(xtest)

wt = float(input("Enter the weight\n"))
ct = float(input("Enter the Cut( 0-4)"))
cl = float(input("Enter the color(0-6)"))
dt = float(input("Enter the Depth\n"))

xnew = [[wt, ct, cl, dt]]
ynew = model.predict(xnew)
print("Diamond Price", ynew[0])

print("Mean sqaured Error", mean_squared_error(ytest, yprd))
print("Variance Score", r2_score(ytest, yprd))
print("Accuracy", model.score(x, y) * 100)

#plt.scatter(data.loc[:,'carat'].values,data.loc[:,'price'].values)
plt.scatter(data.loc[:,'color'].values,data.loc[:,'price'].values)
plt.title("cut Vs Price ")
plt.xlabel("color")
plt.ylabel("Price ")
plt.show()
