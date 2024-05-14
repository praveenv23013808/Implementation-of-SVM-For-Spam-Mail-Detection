# EX 09-Implementation-of-SVM-For-Spam-Mail-Detection
## DATE: 27-03-2024
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:  
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### Data.info():

![image](https://github.com/Darkwebnew/Implementation-of-SVM-For-Spam-Mail-Detection/assets/143114486/63fdec86-8799-41b9-a728-afad4e1ba13a)

### Accuracy:

![image](https://github.com/Darkwebnew/Implementation-of-SVM-For-Spam-Mail-Detection/assets/143114486/565701d0-f760-48a8-bd83-2026058e9a7b)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
