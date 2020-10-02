import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Data.csv')

x = data[['SAT','Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
reg.fit(x,y)

pickle.dump(reg, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
model.predict([[1000,1]])
