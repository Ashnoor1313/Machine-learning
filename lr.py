import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns = cancer.feature_names)

print(df.info())

print(df.describe())

df['label'] = cancer.target

print(df.head())

X = df.drop('label' ,axis = 1 )

y = df.label

print(X)

print(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25 , random_state = 20)

from sklearn.linear_model import LogisticRegression 

LR = LogisticRegression()

LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)

print(y_pred)

sns.countplot(df['label'])
plt.show()


from sklearn.metrics import classification_report , accuracy_score , confusion_matrix , recall_score , precision_score , f1_score

print(accuracy_score(y_pred , y_test))

print(classification_report(y_pred, y_test))

cm = confusion_matrix(y_pred,y_test)

sns.heatmap(cm , annot= True)
plt.show()


print(recall_score(y_pred,y_test))

print(f1_score(y_pred,y_test))

print(precision_score(y_pred,y_test))
