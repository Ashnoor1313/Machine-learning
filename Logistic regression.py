import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.datasets import fetch_california_housing 

housing = fetch_california_housing (as_frame = True)

df = pd.DataFrame(housing['data'])
print(df)

print(df.head())

print(df.tail())

df['price'] = housing['target']

print(df)

X = df.drop('price',axis = 1)

y = df['price']

print(X)

print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25 , random_state = 42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test) 

print(y_pred) 

from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

print(mean_absolute_error(y_pred,y_test))

print(mean_squared_error(y_pred,y_test))

print(r2_score(y_pred, y_test))


print(model.predict([[8.7,15,7,2,455,2.54,38.98,-122.32]]))


corr = df.corr()

print(sns.heatmap(corr , annot = True))



