import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
df = pd.read_csv("profiles.csv")

#print(df.last_online.head())
#print(df.columns)
print("nnumber of categories:",df.sign.nunique())
print("categories:", df.sign.unique())

df['signsCleaned'] = df.sign.str.split().str.get(0)
print("nnumber of categories:",df.signsCleaned.nunique())
print("categories:", df.signsCleaned.unique())

df.signsCleaned.value_counts()
sns.displot(data=df, x="age", kind="hist", binwidth = 5);
sns.displot(data=df, x="age", hue="sex", kind="hist", binwidth = 5, multiple = "stack");
sns.displot(data=df, x="height", kind="hist", binwidth = 2);
sns.displot(data=df, x="height",hue="sex", kind="hist", binwidth = 2, multiple = "stack");
sns.displot(data=df, x="income",hue="sex", kind="hist", binwidth = 50000, multiple = "stack");
sns.countplot(data=df, y="sex");
sns.countplot(data=df, y="body_type");
sns.countplot(data=df, y="body_type", hue = "sex");
sns.countplot(data=df, y="diet");
sns.countplot(data=df, y="drinks");
sns.countplot(data=df, y="drugs");
#plt.figure(figsize=(6,7))

sns.countplot(data=df, y="education");
sns.countplot(data=df, y="job");
sns.countplot(data=df, y="offspring");
sns.countplot(data=df, y="orientation");
sns.countplot(data=df, y="orientation", hue = "sex");
sns.countplot(data=df, y="pets");
# set figure size
#plt.figure(figsize=(6,7))
sns.countplot(data=df, y="religion");
df['religionCleaned'] = df.religion.str.split().str.get(0)
sns.countplot(data=df, y="religionCleaned");
sns.countplot(data=df, y="signsCleaned");\
sns.countplot(data=df, y="smokes");
sns.countplot(data=df, y="status");
#df.isnull().sum()
cols = ['body_type', 'diet', 'orientation', 'pets', 'religionCleaned',
       'sex', 'job', 'signsCleaned']
df = df[cols].dropna()
df.shape
for col in cols[:-1]:
    df = pd.get_dummies(df, columns=[col], prefix = [col])
print(df.head())   
df.signsCleaned.value_counts()
col_length = len(df.columns)

#Y is the target column, X has the rest
X = df.iloc[:, 1:col_length]
Y = df.iloc[:, 0:1]

#Validation chunk size
val_size = 0.25

#Split the data into chunks
from sklearn.model_selection import train_test_split 
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, random_state = 0)
Y_train = Y_train.to_numpy().ravel()
Y_val = Y_val.to_numpy().ravel()

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
lr_model = LogisticRegression(multi_class="multinomial").fit(X_train, Y_train)
lr_predictions = lr_model.predict(X_train)
from sklearn.metrics import classification_report
print(classification_report(Y_train, lr_predictions))
knn_model = KNeighborsClassifier(n_neighbors = 5).fit(X_train, Y_train)
knn_predictions = knn_model.predict(X_train)

print(classification_report(Y_train, knn_predictions))
cart_model = DecisionTreeClassifier().fit(X_train, Y_train) 
cart_predictions = cart_model.predict(X_train) 
print(classification_report(Y_train, cart_predictions))
from sklearn.metrics import confusion_matrix 
cart_cm = confusion_matrix(Y_train, cart_predictions)
cart_labels = cart_model.classes_
plt.figure(figsize=(10,7))

ax= plt.subplot()
sns.heatmap(cart_cm, annot=True, ax = ax,fmt="d");

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');
ax.yaxis.set_tick_params(rotation=360)
ax.xaxis.set_tick_params(rotation=90)

ax.xaxis.set_ticklabels(cart_labels); 
ax.yaxis.set_ticklabels(cart_labels);
cart_model.get_depth()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
results = cross_val_score(cart_model, X_train, Y_train, cv=kfold, scoring='accuracy')

print(results)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

cart_model20 = DecisionTreeClassifier(max_depth = 20).fit(X_train, Y_train) 
cart_predictions20 = cart_model20.predict(X_train) 
print(classification_report(Y_train, cart_predictions20))
results20 = cross_val_score(cart_model20, X_train, Y_train, cv=kfold, scoring='accuracy')

print(results20)
print("Baseline: %.2f%% (%.2f%%)" % (results20.mean()*100, results.std()*100))
knn_predictionsVal = knn_model.predict(X_val) 
print(classification_report(Y_val, knn_predictionsVal))
final_cm = confusion_matrix(Y_val, knn_predictionsVal)
knn_labels = knn_model.classes_

plt.figure(figsize=(10,7))

ax= plt.subplot()
sns.heatmap(final_cm, annot=True, ax = ax, fmt="d");

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');
ax.yaxis.set_tick_params(rotation=360)
ax.xaxis.set_tick_params(rotation=90)

ax.xaxis.set_ticklabels(knn_labels); 
ax.yaxis.set_ticklabels(knn_labels);