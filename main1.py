import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC 
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pickle

heartdata = pd.read_csv("BNG_heart-statlog.csv")
X = heartdata.drop('class', axis=1)  
y = heartdata['class']  
X = X.loc[0:1000, :]
y = y.loc[0:1000]
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
#svclassifier = SVC(kernel='linear')
svclassifier = SVC(kernel='poly', degree=3)  
#svclassifier = SVC(kernel='rbf')
#svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  
print("PolySVC accuracy : ",accuracy_score(y_test, y_pred, normalize = True))
# save the model to disk
filename = 'finalized_model1.sav'
pickle.dump(svclassifier, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred = loaded_model.predict(X_test)  
print("PolySVC accuracy : ",accuracy_score(y_test, y_pred, normalize = True))