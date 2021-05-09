

import joblib
from sklearn import datasets
import numpy as np
import pandas


from sklearn.neural_network._multilayer_perceptron import MLPClassifier


dataset = datasets.fetch_openml("mnist_784")
X = np.array(dataset.data)  #Our Features
y = np.array(dataset.target) #Our labels

X =  X.astype('float32') 

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]



X_train = X_train /255
X_test = X_test /255



mlp = MLPClassifier(hidden_layer_sizes=(240), max_iter=500, verbose=True)


mlp.fit(X_train, y_train)



print("Training set score: %f" % mlp.score(X_train, y_train)) #output : 0.99
print("Test set score: %f" % mlp.score(X_test, y_test))     #output :0.98


joblib.dump(mlp, "model.pkl")
