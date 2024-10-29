import numpy as np
import pandas as pd

#setting for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#reading the dataset
df = pd.read_csv("biomechanical_specs.csv")
print(df.head(2)) #check the df

print(df.info()) #checking for missing values and types

#plotting the count of classes
sns.countplot(df["class"])
plt.tight_layout()
plt.show()


#converting strings to 1/0 in class column
df["class"] = [1 if each == "Abnormal" else 0 for each in df["class"]]
print(df.head(4)) #check the updated df

print(df.info()) #checking for missing values and types


y = df["class"].values #putting values of classes into y
x_data = df.drop(["class"], axis=1) #splitting labels from data

#plots
sns.pairplot(x_data)
plt.show()


#correlation
corr_matrix = x_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heat Maps')
plt.xticks(rotation = 45)
plt.show()


#normalization
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=42)

#transpose
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ",y_test.shape)

#linear-regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)

test_valid = lr.score(x_test.T, y_test.T)
print("Test validation: {}".format(test_valid))


#knn model
from sklearn.neighbors import KNeighborsClassifier
k = 4
knn = KNeighborsClassifier(n_neighbors= k)
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)
print("{} KNN Test Validation : {}".format(k,knn.score(x_test.T, y_test.T)))

#finfing best n number for knn
score_list = []
for each in range (1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train.T, y_train.T)
    score_list.append(knn2.score(x_test.T, y_test.T))

plt.plot(range(1,15), score_list)
plt.xlabel("k value")
plt.xlabel("Validation")
plt.title("outputs of k values")
plt.show()

#support vector machine
from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train.T, y_train.T)
print("SVM Model Test Validation: {}".format(svm.score(x_test.T, y_test.T)))

#DT Model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train.T, y_train.T)
print("DT Test Validation: {}".format(dt.score(x_test.T, y_test.T)))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 100, random_state=1) #100 trees
rf.fit(x_train.T, y_train.T)
print("RFC Test Validation: {}".format(rf.score(x_test.T, y_test.T)))




