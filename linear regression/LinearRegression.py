import pandas as pd
import numpy as np


df_train  = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

x_train =df_train['x']
y_train =df_train['y']

x_test=df_train['x']
y_test=df_train['y']


x_train =np.array(x_train)
y_train =np.array(y_train)

x_test =np.array(x_test)
y_test =np.array(y_test)



x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)



from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


clf = LinearRegression(normalize =True)

clf.fit(x_train,y_train)

y_pred= clf.predict(x_test)




print clf.predict([[123]])



