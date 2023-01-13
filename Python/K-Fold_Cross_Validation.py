# -*- coding: utf-8 -*-

"""
Created on Thu Nov 4 17:28:23 2021
@author: Jeremy Rodriguez
"""

from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import StratifiedKFold


def verifynSplits(n_splits, X):
  if n_splits>len(X):
    raise ValueError("El numero de particiones no puede ser mayor que el numero de muestras")
  if not isinstance(n_splits, int):
    raise ValueError("El número de particiones debe ser de tipo entero")
  if not n_splits>1:
    raise ValueError("El numero de particiones debe ser mayor que uno '1'")


def StratifiedKfCV(X,y,n_splits):
  index= np.arange(len(X))
  itrs=iteracionestest(X, y, n_splits)
  for test_index in itrs:
    train_index = index[np.logical_not(test_index)]
    test_index = index[test_index]
    yield train_index, test_index


def detType(y):
  if len(np.unique(y))>2:
    return "Multiclase"
  else:
    return "Binaria"


def particionestest(X,y,tipo,n_splits):
  _, idx, idx_i=np.unique(y, return_index=True, return_inverse=True)
  _, classes=np.unique(idx, return_inverse=True)
  y_encoded=classes[idx_i]
  
  if tipo=="Binaria":
    n_clase=2
  else:
    n_clase= len(idx)
    
# y_count = np.bincount(y_encoded)
  y_order = np.sort(y_encoded)
  print(y_order)
  
  allocation = np.asarray(
    [
      np.bincount(y_order[i :: n_splits], minlength=n_clase)
      for i in range(n_splits)
    ]
  )
  
  print(allocation)
  test_folds = np.empty(len(y), dtype="i")
  for k in range(n_clase):
    folds_for_class = np.arange(n_splits).repeat(allocation[:, k])
    print(folds_for_class)
    test_folds[y_encoded == k] = folds_for_class
    print(test_folds)
  return test_folds


def iteracionestest(X,y,n_splits):
  tipo=detType(y)
  test_folds = particionestest(X,y,tipo,n_splits)
  for i in range(n_splits):
    yield test_folds == i
    
    
# Modelo a evaluar
reg= linear_model.LinearRegression()
sumE=0
sumr=0

#Datos
datos=datasets.load_breast_cancer()
X=np.array(datos.data)
y=np.array(datos.target)
# X, y = np.ones((20, 1)), np.hstack(([0] * 15, [1] * 5))
n_splits=3 #valor de K
# X=np.ones((20, 1))
# y=np.array([0,0,0,0,0,1,1,1,2,1,0,0,0,0,0,3,1,1,1,1])
verifynSplits(n_splits, X)
for train, test in StratifiedKfCV(X,y,n_splits):
  print('train - {} | test - {}'.format(
  np.bincount(y[train]), np.bincount(y[test])))
  # X_train, X_test = X[train], X[test]
  # y_train, y_test = y[train], y[test]
  # reg.fit(X_train, y_train)
  # y_pred=reg.predict(X_test)
  # error=mean_squared_error(y_test,y_pred)
  # sumE+=error
  # r2= r2_score(y_test,y_pred)
  # sumr+=r2
  # ErrorT=sumE/n_splits
# r2T= sumr/n_splits

# sumE=0
# sumr=0

# skf = StratifiedKFold(n_splits=n_splits)
# for train, test in skf.split(X, y):
  # X_train, X_test = X[train], X[test]
  # y_train, y_test = y[train], y[test]
  # reg.fit(X_train, y_train)
  # y_pred=reg.predict(X_test)
  # error=mean_squared_error(y_test,y_pred)
  # sumE+=error
  # r2= r2_score(y_test,y_pred)
  # sumr+=r2
# Error=sumE/n_splits
# r2= sumr/n_splits

# print("\nError promedio sin librerias: ",ErrorT)
# print("Valor de r^2 promedio sin librerias: ", r2T)
# print("\nError promedio librerias: ",Error)
# print("Valor de r^2 promedio librerias: ", r2)
