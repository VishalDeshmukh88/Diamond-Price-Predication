
#import all modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn import preprocessing, metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
#----------------------------------------------------------
#create GUI
window=Tk()
window.title("Diamond Price Predication ")
window.geometry('400x300')

title=Label(window,text="Diamond Price Predication ").grid(row=0,column=2)

carat=Label(window,text="Carat(Weight) \t ").grid(row=1,column=2)
weight=Entry(window)
weight.grid(row=1,column=3)

cutlabel=Label(window,text="Cut (Ideal=0 to Fair=4) \t\t ").grid(row=2,column=2)
cut=Entry(window)
cut.grid(row=2,column=3)


colorlabel=Label(window,text="Color (D=0 to J=6)  \t ").grid(row=3,column=2)
color=Entry(window)
color.grid(row=3,column=3)

depth=Label(window,text="Depth \t ").grid(row=4,column=2)
depthh=Entry(window)
depthh.grid(row=4,column=3)

#----------------------------------------------------------
#----------------------------------------------------------
#Read data and sepearte it

data=pd.read_csv("diamonds.csv")
print(data.head())

label_encoder=preprocessing.LabelEncoder()
data['cut']=label_encoder.fit_transform(data['cut'])
data['color']=label_encoder.fit_transform(data['color'])

print(data['cut'].unique())
print(data['color'].unique())


x=data.loc[:,["carat","cut","color","depth"]].values
y=data.loc[:,"price"].values

#----------------------------------------------------------

#----------------------------------------------------------
#split the dataset into train and test and train model

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20)
print("Random Forest Regression ")
Random_model = RandomForestRegressor(n_estimators=10, random_state=42)
Random_model.fit(xtrain, ytrain)
yprd=Random_model.predict(xtest)
print("Mean sqaured Error", mean_squared_error(ytest, yprd))
print("Variance Score", r2_score(ytest, yprd))
print("Accuracy", Random_model.score(x, y) * 100)

def clicked():
    xnew=np.array([[weight.get(),cut.get(),color.get(),depthh.get()]],dtype=float)
    ynew=Random_model.predict(xnew)
    messagebox.showinfo("Diamond Price",("Price:$\n",ynew[0]))

check_btn=Button(window,text="Check Price\n",command=clicked).grid(row=5,column=3)


plt.scatter(ytest,yprd,color='blue')

#plt.plot(ytest,yprd,color='green')

plt.title("Random Forest Regression \n")
plt.xlabel("Features  ")
plt.ylabel("Price")
plt.show()
window.mainloop()
#----------------------------------------------------------

#----------------------------------------------------------
#Regression
print("Linear Regression \n")
Linear_regressor=LinearRegression()
Linear_regressor.fit(xtrain,ytrain)
yprd=Linear_regressor.predict(xtest)

#print all parameter of Linear Regression

print("Mean sqaured Error",mean_squared_error(ytest,yprd))
print("Variance Score",r2_score(ytest,yprd))
print("Coefficent",Linear_regressor.coef_)
print("Intercept ",Linear_regressor.intercept_)
print("Accuracy",Linear_regressor.score(x,y)*100)

#----------------------------------------------------------

#----------------------------------------------------------
#KNN regression
print("Knn Algorithm \n")
errvalue = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    KNNmodel = KNeighborsRegressor(n_neighbors = K)
    KNNmodel.fit(xtrain, ytrain)
    yprd=KNNmodel.predict(xtest)
    error = np.sqrt(mean_squared_error(ytest,yprd))
    errvalue.append(error)
print("Error of KNN",errvalue)
#----------------------------------------------------------

#----------------------------------------------------------
#Decision Tree Regressor

print("Decision Tree Regression ")

DTregressor=DecisionTreeRegressor()
DTregressor.fit(xtrain, ytrain)
yprd = DTregressor.predict(xtest)

print("Mean Absolute Error", metrics.mean_absolute_error(ytest, yprd))
print("Mean Squared Error" ,metrics.mean_squared_error(ytest, yprd))
print("Root mean Squared Error ", np.sqrt(metrics.mean_squared_error(ytest, yprd)))


#----------------------------------------------------------

#----------------------------------------------------------
#SVR Regressor
print("SVR Regression ")
SVRmodel=SVR(kernel="linear",gamma="auto")
SVRmodel.fit(xtrain,ytrain)

print("Mean sqaured Error",mean_squared_error(ytest,yprd))
print("Variance Score",r2_score(ytest,yprd))
print("Coefficent",SVRmodel.coef_)
print("Intercept ",SVRmodel.intercept_)
print("Accuracy",SVRmodel.score(x,y)*100)

#----------------------------------------------------------