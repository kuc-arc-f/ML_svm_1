# -*- coding: utf-8 -*-

# SVM
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer = load_breast_cancer()
#print(cancer.data[:10] )
#print(cancer.target.shape )
#print( type(cancer.target) )
ds =pd.DataFrame(cancer.data )

#quit()
#
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify = cancer.target, random_state=50)

print(X_train.shape, X_test.shape )
print(y_train.shape ,y_test.shape  )
#quit()
model = LinearSVC()
clf = model.fit(X_train,y_train)
#pred= model.predict(X_train)
pred= model.predict(X_test )
#print("pred=", pred.shape )

df= pd.DataFrame(pred)
#print(df.head() )
#quit()
print("train:",clf.__class__.__name__ ,clf.score(X_train,y_train))
print("test:",clf.__class__.__name__ , clf.score(X_test,y_test))

