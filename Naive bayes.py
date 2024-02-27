import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


df = pd.read_csv("D:\practice\iris.csv")

print(df)

print(df.info())

print(df.isna().sum)

print(df.describe())

from sklearn.model_selection import train_test_split

X = df.drop('species', axis = 1)

y = df['species']

X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size= 0.5, random_state=42)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()

NB.fit(X_train,y_train)

y_pred = NB.predict(X_test)
print(y_pred)

from sklearn.metrics import classification_report , accuracy_score , confusion_matrix

print(classification_report(y_pred,y_test))

print(accuracy_score(y_pred,y_test))

cm = confusion_matrix(y_pred,y_test)

sns.heatmap(cm,annot=True)
plt.show()