# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary python packages using import statements.

2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3. Split the dataset using train_test_split.

4. Calculate Y_Pred and accuracy.

5. Print all the outputs.

6. End the Program. 

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Javan Rufus J
RegisterNumber: 212224230104

```
```
import chardet
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Dataset location
path = r"C:\Users\admin\Downloads\spam.csv"

# Detect encoding type
with open(path, "rb") as rawdata:
    encoding_result = chardet.detect(rawdata.read(100000))

print(encoding_result)

# Read dataset
data = pd.read_csv(
    path,
    encoding="Windows-1252"
)

# Display first records
print(data.head())

# Dataset details
print(data.info())

# Null value checking
print(data.isnull().sum())

# Selecting required columns
x = data["v2"]
y = data["v1"]

# Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

# Text conversion
vectorizer = CountVectorizer()

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# SVM classifier
svc = SVC()

# Training model
svc.fit(x_train, y_train)

# Prediction
y_pred = svc.predict(x_test)

print(y_pred)

# Accuracy calculation
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
```

## Output:

CODE : 

<img width="827" height="512" alt="image" src="https://github.com/user-attachments/assets/e4b8b86e-a69f-4a7c-bd86-6834dcdbf886" />

<img width="825" height="475" alt="image" src="https://github.com/user-attachments/assets/bb29c2e6-e801-48ec-a4ec-f9f478117218" />

RECIEVED : 

<img width="1056" height="685" alt="image" src="https://github.com/user-attachments/assets/a6618c1c-d3c0-4ad5-a88f-a7b74ab13c01" />



## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
