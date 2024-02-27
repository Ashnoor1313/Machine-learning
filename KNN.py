import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("D:\practice\iris.csv")

print(df)

print(df.info())

print(df.describe())

print(df['species'].value_counts())

sns.pairplot(df , hue = 'species')
plt.show()

sns.countplot(x = df.species)
plt.show()


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

print(df['species'].value_counts())

corr = df.corr()
sns.heatmap(corr , annot = True)
plt.show()

X = df.drop('species', axis = 1)

y = df['species']

print(X)

print(y)

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.5 , random_state= 42)

from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=3)

'''knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix , accuracy_score

print(classification_report(y_pred,y_test))


cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm , annot=True)
plt.show()

print(accuracy_score(y_pred,y_test))

predction = knn.predict([[5.3,3.6,1.3,1.2]])

print([predction])'''



from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(y_pred)

from sklearn.metrics import classification_report, confusion_matrix , accuracy_score

print(classification_report(y_pred,y_test))

cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm , annot=True)
plt.show()

print(accuracy_score(y_pred,y_test))

predction = model.predict([[5.3,3.6,1.3,1.2]])

print([predction])