#importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix,classification_report)
import seaborn as sb
import matplotlib.pyplot as plt

#reading csv
df=pd.read_csv('Student Depression Dataset.csv')

#data exploration
print(df.head())

print(df.shape)

print(df.info)

print(df.dtypes)

#plotting
plt.figure(figsize=(5,5))
sb.boxplot(x='Depression', y='Work/Study Hours', data=df,color='pink')
plt.title("Study Hours vs Depression")
plt.show()

plt.figure(figsize=(5,5))
sb.countplot(x='Depression', data=df,color='yellow')
plt.show()

#feature scaling and feature engineering
print(df.isnull().sum())

df=df.drop(columns='id')
df=df.drop(columns='City')
df=df.drop(columns='Profession')
df=df.drop(columns='Job Satisfaction')
df=df.drop(columns='Work Pressure')

df = df.dropna(subset=['Financial Stress'])

Ss=StandardScaler()
Le=LabelEncoder()

df=pd.get_dummies(df,columns=['Dietary Habits','Degree','Sleep Duration'],drop_first=True)
df['Gender']=Le.fit_transform(df['Gender'])
df['Have you ever had suicidal thoughts ?']=Le.fit_transform(df['Have you ever had suicidal thoughts ?'])
df['Family History of Mental Illness']=Le.fit_transform(df['Family History of Mental Illness'])

#initializing x and y
y=df['Depression']
x=df.drop(columns='Depression')

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#scaling numeric columns
num=['Age','CGPA','Academic Pressure','Study Satisfaction','Work/Study Hours','Financial Stress']

x_train[num]=Ss.fit_transform(x_train[num])
x_test[num]=Ss.transform(x_test[num])

#logistic regression 
logistic=LogisticRegression()
logistic.fit(x_train,y_train)
y_log=logistic.predict(x_test)
log_acc=accuracy_score(y_test,y_log)*100
print("logistic regression accuracy=",log_acc)

#random forest classifier
Rf=RandomForestClassifier(random_state=42,n_estimators=100)
Rf.fit(x_train,y_train)
y_rf=Rf.predict(x_test)
rf_acc=accuracy_score(y_test,y_rf)*100
print("random forest classifier accuracy=",rf_acc)


#confusion matrix
confusion = confusion_matrix(y_test, y_log)
plt.figure(figsize=(5,5))
sb.heatmap(confusion, annot=True, fmt='d', cmap='pink')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of logistic regression")
plt.show()


confusion = confusion_matrix(y_test, y_rf)
plt.figure(figsize=(5,5))
sb.heatmap(confusion, annot=True, fmt='d', cmap='pink')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of random forest classifier")
plt.show()

models = ['Logistic Regression','Random Forest']
accuracy = [log_acc, rf_acc]

plt.figure(figsize=(5,5))
plt.bar(models, accuracy,color='red')
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()

#classification report
print("classification report of logistic regression",classification_report(y_test,y_log))

print("classification report of random forest classifier",classification_report(y_test,y_rf))
