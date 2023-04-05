# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:212222240081
RegisterNumber:RAGUNATH R

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:

## df.head
![image](https://user-images.githubusercontent.com/113915622/230003644-a6c8175c-2957-446f-9385-753563c37389.png)

## df.tail
![image](https://user-images.githubusercontent.com/113915622/230002671-0141a673-e817-4369-864e-44ed6a01e2b2.png)

# #Array value of X
![image](https://user-images.githubusercontent.com/113915622/230002749-aa3741e9-c8fa-4acd-8fe2-3cb807a57d09.png)

## Array value of Y
![image](https://user-images.githubusercontent.com/113915622/230002869-92b558f0-dd55-4d06-ace3-53965de41215.png)

## Values of Y prediction
![image](https://user-images.githubusercontent.com/113915622/230002985-9742f21e-db65-40af-ab25-a17911ac6b4c.png)

## Array values of Y test
![image](https://user-images.githubusercontent.com/113915622/230003059-99be4446-3b70-4ca3-8522-ae7be0f5071d.png)

## Training Set Graph
![image](https://user-images.githubusercontent.com/113915622/230003157-6db35c24-28a2-4411-9252-f658409af75c.png)

## Test set graph
![image](https://user-images.githubusercontent.com/113915622/230003242-4e98fb4c-13ab-4b50-85e0-03bc9461d505.png)

## Values of MSE, MAE and RMSE
![image](https://user-images.githubusercontent.com/113915622/230003341-ce9b56c4-757e-4151-89ee-f1e2cd050923.png)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
