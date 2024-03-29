# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vineela Shaik
RegisterNumber:  212223040243
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

### 1.PLACEMENT DATA:  
![Screenshot 2024-03-29 113250](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/2805e64e-52d9-49e8-aec8-59133dd9acb6)
### 2.SALARY DATA:
![Screenshot 2024-03-29 113306](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/26d11fe3-a1b9-4c77-9d7d-5505de50c9a0)
### 3.CHECKING THE NULL() FUNCTION:
![Screenshot 2024-03-29 113329](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/9b9e4e2d-e5d1-4cad-96b1-23763422948f)
### 4.DATA DUPLICATE:
![Screenshot 2024-03-29 113342](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/dcb4b902-8204-4ca2-84c7-7416b94829f8)
### 5.PRINT DATA:
![Screenshot 2024-03-29 113355](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/dc12d481-8361-4627-8f8e-562a45d98704)
### 6.DATA STATUS:
![Screenshot 2024-03-29 113407](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/d7b2176a-87ad-4661-81ba-403a2fdc30de)
![Screenshot 2024-03-29 113416](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/cfdfc0e2-ce3d-45f5-b3f5-2702f4df680c)
### 7.Y_PREDICTION ARRAY:
![Screenshot 2024-03-29 113429](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/80e5d893-926f-4a2a-8bb7-d8fc1858749d)

### 8.ACCURACY VALUE:
![Screenshot 2024-03-29 113436](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/c6c42e8b-aecd-4902-b19f-5b281354f0d6)
### 9.CONFUSION ARRAY:
![Screenshot 2024-03-29 113518](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/ee7f0b0c-aca8-41bc-975d-fca433a33fdf)
![Screenshot 2024-03-29 113544](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/dfdde437-e9bc-46f7-8f8e-c969342daadc)
### 10.CLASSIFICATION REPORT:
![Screenshot 2024-03-29 113600](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/c0890097-711b-46cb-9206-57678c2fb97c)
### PREDICTION OF LR:
![Screenshot 2024-03-29 113612](https://github.com/VineelaShaik/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144340862/38ff348e-d19a-4655-ac6b-145e428eaa43)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
