#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:56:34 2019

@author: Montserrat Rodríguez Zamorano
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from time import time
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

#para evitar los warnings en logisticRegression
import warnings
warnings.filterwarnings("ignore")

########################### TRATAMIENTO DE DATOS #############################

#Función para cargar y procesar los datos

def loadData(filedata):
    print('Loading data from file ', filedata, '...')
    #convertir las ? en no  asignados
    cleandata = pd.read_csv(filedata, header =None, na_values=['   ?', '     ?', '?'])
    print('Done')
    x, y = separateLabels(cleandata)
    #escribir los tamaños de los datos
    print('x shape:', x.shape)
    print('y shape:', y.shape)
    
    input('\n--- Tap enter to continue ---\n')
    
    return x,y

#Función para tratar los valores perdidos
def convertMissingValues(data):
    #sustituir valores nan por la media de los valores de la columna
    imp = Imputer(missing_values=np.nan, strategy='mean')
    imp.fit(data)
    impdata = imp.transform(data)
    return impdata

#Funcion para normalizar datos
def normalizeData(train, test):
    norm = preprocessing.Normalizer()
    train = norm.fit(train)
    train = norm.transform(train)
    test = norm.transform(test)
    return train, test

#Función para estandarizar los datos    
def standarizeData(train, test):
    sc = StandardScaler()
    #ajustar con datos de train
    sc.fit(train)
    #usar los mismos parametros de train en test
    train_ = sc.transform(train)
    test_ = sc.transform(test)
    
    return train_, test_
    
#Función para seleccionar características    
def featureSelection(train, test, percentage):
    pca = PCA(percentage)
    #ajustar con datos de train
    pca.fit(train)
    train_ = pca.transform(train)
    #usar los mismos parametros de train en test
    test_ = pca.transform(test)
    return train_, test_

#Función para dividir el conjunto de datos en train y test    
def setTrainTest(x,y, testsize):
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=testsize, random_state=0)
    return X_train, X_test, y_train, y_test

#Función para separar etiquetas de atributos  
    
def separateLabels(data):
    array_data = data.values
    #guardamos los atributos
    x = np.array(array_data[:,0:1558], np.float64)
    #guardamos las etiquetas
    y = np.array(array_data[:,1558])
    return x,y

#############################################################################
#############################################################################

#Funciones para ajustar el modelo, realizar predicciones y calcular accuracy
    
def adjustModel(xtrain, ytrain, xtest, ytest, model):
    #ajustar el modelo
    model.fit(xtrain, ytrain)
    #realizar predicciones
    y_pred = model.predict(xtest)
    #calcular accuracy
    accuracy = metrics.accuracy_score(ytest, y_pred)
    return y_pred, accuracy

######################### REGRESIÓN LOGÍSTICA #############################
    
#función para validación cruzada en regresión logística  
def crossValidationLR(x, y, numsplits, hyperparameters):
    lr = LogisticRegression(solver='lbfgs', random_state=0)
    hyp = {'C': hyperparameters}
    gr = GridSearchCV(estimator=lr,param_grid=hyp, cv=numsplits)
    gr.fit(x,y)
    bestc = gr.best_params_.get('C')
    bestmean = gr.best_score_
    listofmeans = gr.cv_results_['mean_test_score']

    return bestc, bestmean, listofmeans

def logisticRegression(xtrain, ytrain, xtest, ytest, c):
    #seleccionamos como modelo regresion logistica
    model = LogisticRegression(C=c, solver='lbfgs')
    #ajustar el modelo, devuelve predicciones y score
    predictions, accuracy = adjustModel(xtrain, ytrain, xtest, ytest, model)
    return predictions, accuracy

################################# SVM #######################################

#función para validación cruzada en support vector machine con nucleo gaussiano
def crossValidationSVM(x, y, numsplits, hyperparameters):
    svm_ = svm.SVC(kernel='rbf',max_iter = 500, random_state=0)
    hyp = {'C': hyperparameters}
    gr = GridSearchCV(estimator=svm_,param_grid=hyp, cv=numsplits)
    gr.fit(x,y)
    bestc = gr.best_params_.get('C')
    bestmean = gr.best_score_
    listofmeans = gr.cv_results_['mean_test_score']

    return bestc, bestmean, listofmeans

def SVM(xtrain, ytrain, xtest, ytest, c):
    #seleccionamos como modelo svm
    model = svm.SVC(C=c, kernel='rbf', max_iter = 500, random_state=0)
    #ajustar el modelo, devuelve predicciones y score
    predictions, accuracy = adjustModel(xtrain, ytrain, xtest, ytest, model)
    return predictions, accuracy
############################ RANDOM FOREST ##################################
    
#función para validación cruzada en random forest 
def crossValidationRF(x, y, numsplits, hyperparameters):    
    rf = RandomForestClassifier(bootstrap=True, max_features='sqrt', random_state=0)
    hyp = {'n_estimators': hyperparameters}
    gr = GridSearchCV(estimator=rf,param_grid=hyp, cv=numsplits)
    gr.fit(x,y)
    best_est = gr.best_params_.get('n_estimators')
    bestmean = gr.best_score_
    listofmeans = gr.cv_results_['mean_test_score']

    return best_est, bestmean, listofmeans
    
def randomForest(xtrain, ytrain, xtest, ytest, estimators):
    #seleccionamos como modelo random forest
    model = RandomForestClassifier(n_estimators=estimators, bootstrap=True, max_features='sqrt', random_state=0)
    #ajustar el modelo
    predictions, accuracy = adjustModel(xtrain, ytrain, xtest, ytest, model)
    return predictions, accuracy

#############################################################################
#############################################################################
    

###################### ESTUDIO DEL ERROR ####################################

#Funcion para pintar matriz de confusion
    
def paintConfussionMatrix(ytest, ypred):
    labels = ['ad.', 'nonad.']
    cm = metrics.confusion_matrix(ytest, ypred, labels)
    #tamaño figura
    plt.figure(figsize=(9,9))
    #pintar la matriz
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap=plt.cm.Blues)
    #nombres de los ejes
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confussion matrix')
    plt.show()
    
#############################################################################

#Algunas funciones que sirven para ir ajustando
def testFSNoStandarizationLR(xtrain, xtest, ytrain, ytest):
    print('\nLets see what happens without standarization...\n')
    train_ = np.copy(xtrain)
    test_ = np.copy(xtest)
    print('X_train shape before feature selection:', train_.shape)
    train_, test_ = featureSelection(train_, test_, 0.9)
    print('X_train shape after feature selection:', train_.shape)
    input('\n--- Tap enter to continue ---\n')
    print('Prediction without standarization')

    #seleccionamos como modelo regresion logistica
    model = LogisticRegression(solver='lbfgs')
    #ajustar el modelo, devuelve predicciones y score
    predictions, accuracy = adjustModel(train_, ytrain, test_, ytest, model)

    print('Accuracy: {:.5f}'.format(accuracy*100),'%')
    print('Eout: {:.5f}'.format(1.0-accuracy))
    input('\n--- Tap enter to continue ---\n')

def testNoProcessingLR(xtrain, xtest, ytrain, ytest):
    print('Prediction without data processing')

    #seleccionamos como modelo regresion logistica
    model = LogisticRegression(solver='lbfgs')
    #ajustar el modelo, devuelve predicciones y score
    predictions, accuracy = adjustModel(xtrain, ytrain, xtest, ytest, model)

    print('Accuracy: {:.5f}'.format(accuracy*100),'%')
    print('Eout: {:.5f}'.format(1.0-accuracy))
    input('\n--- Tap enter to continue ---\n')

#############################################################################
#                               MODELOS                                     #
#############################################################################
    

#MODELO 1 (LINEAL): REGRESIÓN LOGÍSTICA
def LRclassification(x,y): 
    print('\n------------------LOGISTIC REGRESSION------------------\n')    
    #separar en training y test
    X_train, X_test, y_train, y_test = setTrainTest(x,y, 0.2)
        
    #transformamos los valores nan
    X_train_ = convertMissingValues(X_train)
    X_test_ = convertMissingValues(X_test)
    
    #qué pasa si no hacemos más procesamiento de datos
    testNoProcessingLR(X_train_, X_test_, y_train, y_test)
    
    #que pasa si seleccionamos caracteristicas sin estandarizar
    testFSNoStandarizationLR(X_train_, X_test_, y_train, y_test)
    
    #estandarizar datos
    print('Data before standarizacion')
    print(np.around(X_train_, decimals = 2))
    input('\n--- Tap enter to continue ---\n')
    
    #estandarizar y visualizar datos
    X_train_, X_test_ = standarizeData(X_train_, X_test_)
    print('Data after standarizacion')
    print(np.around(X_train_, decimals = 2))
    input('\n--- Tap enter to continue ---\n')
    
    #selección de caracterisitcas
    print('X_train shape before feature selection:', X_train_.shape)
    X_train_, X_test_ = featureSelection(X_train_, X_test_, 0.9)
    print('X_train shape after feature selection:', X_train_.shape)
    input('\n--- Tap enter to continue ---\n')
            
    print('Starting logistic regression classification.\n')
    
    #lista de hiperparametros a probar en el modelo
    listh = []
    for exp in range(-3,3):
        listh.append(10**exp)
    
    #tiempo de inicio
    start = time()
    #aplicar cross validation para seleccionar el mejor modelo
    selectedc, selectedacc, listmeans = crossValidationLR(X_train_, y_train, 10, listh)    
    #tiempo de fin
    end = time() - start
    print('Cross validation finished.\nTime used: ', end)
    print('Best c:', selectedc)
    #error con los datos de training
    print('Ein: {:.5f}'.format(1.0-selectedacc))
    input('\n--- Tap enter to continue ---\n')
    
    #pintar la grafica para ver la evolucion del hiperparametro c
    plt.plot(listh, listmeans)
    plt.xlabel('hyperparameter c')
    plt.ylabel('accuracy')
    plt.title('Accuracy-Hyperparameter relation in LR')
    plt.show()
    input('\n--- Tap enter to continue ---\n')
    
    #aplicar regresion logistica con los hiperparametros seleccionados a datos de test
    y_pred, accuracy = logisticRegression(X_train_, y_train, X_test_, y_test, selectedc)
    print('Using testing data...\n')
    print('Accuracy: {:.5f}'.format(accuracy*100),'%')
    
    #mostrar error con datos de test
    print('Eout: {:.5f}'.format(1.0-accuracy))
    input('\n--- Tap enter to continue ---\n')
    
    #pintar matriz de confusion
    paintConfussionMatrix(y_test, y_pred)
    input('\n--- Tap enter to continue ---\n')

#MODELO 2 (NO LINEAL): SVM
def SVMclassification(x,y):
    #separar en training y test
    X_train, X_test, y_train, y_test = setTrainTest(x,y, 0.2)
        
    #transformamos los valores nan
    X_train_ = convertMissingValues(X_train)
    X_test_ = convertMissingValues(X_test)
    
    #estandarizar datos 
    X_train_, X_test_ = standarizeData(X_train_, X_test_)
    
    #lista de hiperparametros a probar en el modelo
    listh = []
    for exp in range(-2,1):
        listh.append(10**exp)
    
    print('Model without PCA')
    print('This is going to take some time...')
    
    #tiempo de inicio
    start = time()
    #aplicar cross validation para seleccionar el mejor modelo
    selectedc, selectedacc, listmeans = crossValidationSVM(X_train_, y_train, 10, listh)    
    #tiempo de fin
    end = time() - start
    print('Cross validation finished.\nTime used: ', end)
    print('Best c:', selectedc)
    #error con los datos de training
    print('Ein: {:.5f}'.format(1.0-selectedacc))
    input('\n--- Tap enter to continue ---\n')
    
    #pintar la grafica para ver la evolucion del hiperparametro c
    plt.plot(listh, listmeans)
    plt.xlabel('hyperparameter c')
    plt.ylabel('accuracy')
    plt.title('Accuracy-Hyperparameter relation in SVM')
    plt.show()
    input('\n--- Tap enter to continue ---\n')
    
    #aplicar svm con los hiperparametros seleccionados a datos de test
    y_pred, accuracy = SVM(X_train_, y_train, X_test_, y_test, selectedc)
    print('Using testing data...\n')
    print('Accuracy: {:.5f}'.format(accuracy*100),'%')
    
    #mostrar error con datos de test
    print('Eout: {:.5f}'.format(1.0-accuracy))
    input('\n--- Tap enter to continue ---\n')
    
    #pintar matriz de confusion
    paintConfussionMatrix(y_test, y_pred)
    input('\n--- Tap enter to continue ---\n')
    
    print('Model with PCA')
    print('This is going to take some time...')
    
    X_train_, X_test_ = featureSelection(X_train_, X_test_, 0.9)
    #tiempo de inicio
    start = time()
    #aplicar cross validation para seleccionar el mejor modelo
    selectedc, selectedacc, listmeans = crossValidationSVM(X_train_, y_train, 10, listh)    
    #tiempo de fin
    end = time() - start
    print('Cross validation finished.\nTime used: ', end)
    print('Best c:', selectedc)
    #error con los datos de training
    print('Ein: {:.5f}'.format(1.0-selectedacc))
    input('\n--- Tap enter to continue ---\n')
    
    #pintar la grafica para ver la evolucion del hiperparametro c
    plt.plot(listh, listmeans)
    plt.xlabel('hyperparameter c')
    plt.ylabel('accuracy')
    plt.title('Accuracy-Hyperparameter relation in SVM')
    plt.show()
    input('\n--- Tap enter to continue ---\n')
    
    #aplicar svm con los hiperparametros seleccionados a datos de test
    y_pred, accuracy = SVM(X_train_, y_train, X_test_, y_test, selectedc)
    print('Using testing data...\n')
    print('Accuracy: {:.5f}'.format(accuracy*100),'%')
    
    #mostrar error con datos de test
    print('Eout: {:.5f}'.format(1.0-accuracy))
    input('\n--- Tap enter to continue ---\n')
    
    #pintar matriz de confusion
    paintConfussionMatrix(y_test, y_pred)
    input('\n--- Tap enter to continue ---\n')
 

#MODELO 3 (NO LINEAL): RANDOM FOREST
def RFclassification(x,y):
    print('\n---------------------RANDOM FOREST---------------------\n')
    #separar en training y test
    X_train, X_test, y_train, y_test = setTrainTest(x,y, 0.2)
    
    #transformamos los valores nan, la funcion no admite nan
    X_train_ = convertMissingValues(X_train)
    X_test_ = convertMissingValues(X_test)
    
    #estandarizar datos 
    X_train_, X_test_ = standarizeData(X_train_, X_test_)

    #selección de caracteristicas importantes
    X_train_, X_test_ = featureSelection(X_train_, X_test_, 0.9)
        
    print('Starting Random Forest classification.\n')
    #hiperparametros a probar en el modelo
    listh = []
    for i in range(1,11):
        listh.append(10*i)
    
    #tiempo de inicio
    start = time()
    #aplicar cross validation para seleccionar el mejor modelo
    selectedest, selectedmean, listmeans = crossValidationRF(X_train_, y_train, 10, listh)    
    #tiempo de fin
    end = time() - start
    print('Cross validation finished.\nTime used: ', end)
    #error con los datos de training
    print('Selected n_estimator: ', selectedest)
    print('Ein: {:.5f}'.format(1.0-selectedmean))
    input('\n--- Tap enter to continue ---\n')
    
    #pintar la grafica para ver la evolucion del hiperparametro c
    plt.plot(listh, listmeans)
    plt.xlabel('hyperparameter n_estimators')
    plt.ylabel('accuracy')
    plt.title('Accuracy-Hyperparameter relation in RF')
    plt.show()
    input('\n--- Tap enter to continue ---\n')
    
    #aplicar random forest con los hiperparametros seleccionados a datos de test
    y_pred, accuracy = randomForest(X_train_, y_train, X_test_, y_test, 100)
    print('Using testing data...\n')
    print('Accuracy: {:.5f}'.format(accuracy*100),'%')
    
    #mostrar error con datos de test
    print('Eout: {:.5f}'.format(1.0-accuracy))
    input('\n--- Tap enter to continue ---\n')
    
    #pintar matriz de confusion
    paintConfussionMatrix(y_test, y_pred)
    input('\n--- Tap enter to continue ---\n')
    

###########################################################################
print('Final project: Internet Ads Classification.') 
print('Student: Montserrat Rodríguez Zamorano\n\n')
rawx, y = loadData('datos/ad.data')
print('Printing some data.')
print(rawx[110:120,:], '\n',y[110:120])
input('\n--- Tap enter to continue ---\n')
LRclassification(rawx,y)
RFclassification(rawx,y)
SVMclassification(rawx,y)
    