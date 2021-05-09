import numpy as np
from sklearn.metrics import classification_report
import joblib
from sklearn import datasets
import pandas


dataset = datasets.fetch_openml("mnist_784")
X = np.array(dataset.data)  #Our Features
y = np.array(dataset.target) #Our labels

X =  X.astype('float32') 

X_test,y_test = X[60000:], y[60000:]
 




X_test = X_test /255

model = joblib.load('model.pkl')


y_pred = model.predict(X_test)

print(classification_report(y_pred,y_test))








