#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRABAJO 3
Nombre Estudiante: Montserrat Rodriguez Zamorano
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
#para evitar los warnings en logisticRegression
import warnings
warnings.filterwarnings("ignore")

############## Problema clasificación ####################

##########################################################
# Funcion para leer los datos
def classification_loadData(filetrain, filetest):
    print('Loading data from file ', filetrain, '...')
    print('Loading data from file ', filetest, '...')
    train = pd.read_csv(filetrain, header =None)
    test = pd.read_csv(filetest, header=None)
    #convertir los valores en array
    array_train = train.values
    array_test = test.values
    #separamos los datos en atributos y salida
    Xtrain = np.array(array_train[:,0:64], np.float64)
    ytrain = np.array(array_train[:,64], np.float64)
    Xtest = np.array(array_test[:,0:64], np.float64)
    ytest = np.array(array_test[:,64], np.float64)

    print('Done.')

    return Xtrain, Xtest, ytrain, ytest

#Funcion para dibujar digitos
def showDigits(x,y,num):
    plt.figure(figsize=(6,2))
    for i in range(num):
        plt.subplot(2,5, i+1)
        plt.imshow(np.split(x[i],8),cmap=plt.cm.gray_r)
        plt.axis('off')
    plt.show()
    #imprimimos las etiquetas
    print(y[0:num])


##########################################################
#Funcion para normalizar datos
def normalizeData(x):
    norm = preprocessing.Normalizer()
    x_ = norm.fit_transform(x)
    return x_

#Funcion para estandarizar datos (media 0 y varianza 1)
def standarizeData(x):
    stand = preprocessing.StandardScaler()
    x_ = stand.fit_transform(x)
    return x_

#Funcion para preprocesar los datos para clasificacion
def classificationPreprocessing(x):
    print('Preprocessing data...')
    x_= normalizeData(x)
    print('Done.')
    return x_

##########################################################

def classificationCrossValidation(x,y, numsplits):
    #rango de exponentes para c
    c_s = range(-3,3)
    #dividimos el conjunto en numsplits subconjuntos
    skf = StratifiedKFold(n_splits=numsplits)
    skf.get_n_splits(x, y)
    bestc = 0
    bestacc = 0
    #probamos diferentes cs
    for i in c_s:
        c = 10**i
        accuracy = 0
        #iteramos en los subconjuntos
        for train_index, test_index in skf.split(x, y):
            #subconjuntos de train y test
            X_train_, X_test_ = x[train_index], x[test_index]
            y_train_, y_test_ = y[train_index], y[test_index]
            model = LogisticRegression(C=c, solver='lbfgs')
            #ajustar el modelo
            model.fit(X_train_, y_train_)
            #predicciones usando el conjunto test
            y_pred = model.predict(X_test_)
            #calcular score
            accuracy += metrics.accuracy_score(y_test_, y_pred)
        #score de media con c actual
        mean_accuracy = accuracy/numsplits
        #si el c actual da mejor resultado en media
        #lo guardamos con el mejor c
        if mean_accuracy > bestacc:
            bestacc = mean_accuracy
            bestc = c
    return bestc, bestacc

##########################################################
#Funciones para ajustar el modelo y realizar predicciones
def adjustModel(xtrain, ytrain, xtest, ytest, model):
    #ajustar el modelo
    model.fit(xtrain, ytrain)
    #realizar predicciones
    y_pred = model.predict(xtest)
    #calcular score
    accuracy = metrics.accuracy_score(ytest, y_pred)
    return y_pred, accuracy

def logisticRegression(xtrain, ytrain, xtest, ytest, c):
    #seleccionamos como modelo regresion logistica
    model = LogisticRegression(C=c, solver='lbfgs')
    #ajustar el modelo, devuelve predicciones y score
    predictions, accuracy = adjustModel(xtrain, ytrain, xtest, ytest, model)
    return predictions, accuracy

##########################################################
#Funcion para pintar matriz de confusion
def paintConfussionMatrix(ytest, ypred, acc):
    cm = metrics.confusion_matrix(ytest, ypred)
    #tamaño figura
    plt.figure(figsize=(9,9))
    #pintar la matriz
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap=plt.cm.Blues)
    #nombres de los ejes
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confussion matrix', size = 10)
    plt.show()

#Funcion para pintar los digitos clasificados erroneamente
def paintError(xtest, ytest, ypred, num):
    print(num, ' misclassified digits representation:')
    #inicializar variables
    ind = 0
    indErroneos = []
    #para cada pareja etiqueta-prediccion, comprobar si son iguales
    for y, pred in zip(ytest, ypred):
     if (y != pred):
         #guardar indice
         indErroneos.append(ind)
     ind +=1
    print('Predicted label:')
    #pintar erroneos
    showDigits(xtest[indErroneos], ypred[indErroneos], num)
    #escribir etiquetas reales
    print('Actual label: ', ytest[indErroneos][:num])

##########################################################

def classificationProblem(filex, filey):
    #cargamos la base de datos
    X_train, X_test, y_train, y_test=classification_loadData(filex, filey)
    #escribir los tamaños de los archivos
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    input('\n--- Tap enter to continue ---\n')
    #representar algunos digitos
    print("10 digits representation:")
    showDigits(X_train, y_train, 10)
    input('\n--- Tap enter to continue ---\n')
    #preprocesar los datos
    xtrain_processed = classificationPreprocessing(X_train)
    xtest_processed = classificationPreprocessing(X_test)
    #mostrar los digitos preprocesados
    print('10 preprocessated digits representation:')
    showDigits(xtrain_processed, y_train, 10)
    input('\n--- Tap enter to continue ---\n')
    #tiempo de inicio
    start = time()
    #aplicar cross validation
    bestc, bestacc = classificationCrossValidation(xtrain_processed, y_train, 10)
    #tiempo de fin
    end = time() - start
    print('Time used in Crossvalidation : ', end)
    print('Best c:', bestc)
    #error con los datos de training
    print('Ein: {:.5f}'.format(1.0-bestacc))
    #aplicar regresion logistica
    y_pred, accuracy = logisticRegression(xtrain_processed, y_train, xtest_processed, y_test, bestc)
    print('Accuracy: {:.2f}'.format(accuracy*100),'%')
    #error con datos de test
    print('Eout: {:.5f}'.format(1.0-accuracy))
    input('\n--- Tap enter to continue ---\n')
    #pintar matriz de confusion
    paintConfussionMatrix(y_test, y_pred, accuracy)
    input('\n--- Tap enter to continue ---\n')
    #pintar digitos erroneos
    paintError(xtest_processed, y_test, y_pred, 10)
    input('\n--- Tap enter to continue ---\n')

##########################################################

############### PROBLEMA REGRESION #####################
#Funcion para cargar los datos de los archivos
def regression_loadData(file):
    print('Loading data from file ', file, '...')
    #inicializar listas de atributos y output
    data_input = []
    data_output = []
    #abrir el archivo
    with open(file, 'r') as cvsfile:
        #ponemos como delimitador la tabulacion
        csvreader = csv.reader(cvsfile, delimiter='\t')
        #para cada fila
        for row in csvreader:
            #guardamos las primeras seis columnas como atributos
            inp = np.array(row[:5])
            data_input.append(inp)
            #guardar la ultima columna como output
            outp = row[5]
            data_output.append(outp)
    #convertir a array
    data_input = np.array(data_input, np.float64)
    data_output = np.array(data_output, np.float64)
    print(data_input.shape)
    print(data_output.shape)
    print('Done.')
    return data_input, data_output

def minMaxScaler(x):
    scaler = preprocessing.MinMaxScaler()
    x_ = scaler.fit_transform(x)
    return x_
def regressionPreprocessing(x):
    print('Preprocessing data...')
    x_ = minMaxScaler(x)
    x_ = standarizeData(x_)
    print('Done.')
    return x_

#Funcion que representa los datos predecidos
def printError(label, pred):
    #pintar predicciones
    plt.scatter(label, pred)
    #pintar recta para ver como de correcta es la prediccion
    plt.plot([100,150], [100,150], color = 'y')
    plt.xlabel('Actual value')
    plt.ylabel('Prediction')
    plt.show()

def ridge_regressionCV(x,y, numsplits):
    alpha_s = range(0,10)
    #dividimos el conjunto en numsplits subconjuntos
    kf = KFold(n_splits=numsplits)
    kf.get_n_splits(x, y)
    bestalpha = 0
    bestmse = 1000
    for i in alpha_s:
        alpha = 10**(-i)
        mse_mean = 0
        mse = 0
        #iteramos por los conjuntos
        for train_index, test_index in kf.split(x, y):
            X_train_, X_test_ = x[train_index], x[test_index]
            y_train_, y_test_ = y[train_index], y[test_index]
            #probamos el modelo con el alpha escogido
            model = Ridge(alpha=alpha)
            model.fit(X_train_, y_train_)
            y_pred = model.predict(X_test_)
            mse += mean_squared_error(y_test_, y_pred)
        mse_mean = mse/numsplits
        #si el alpha nos ha dado un mejor error cuadratico medio que el alpha guardado
        if mse_mean < bestmse:
            bestmse = mse_mean
            #guardar el alpha como mejor alpha hasta el momento
            bestalpha = alpha
    return bestalpha, bestmse

def ridge_regressionProblem(file):
    #cargar datos del archivo file
    x, y = regression_loadData(file)
    input('\n--- Tap enter to continue ---\n')
    #preprocesar datos
    x_preprocessed = regressionPreprocessing(x)
    #separar en conjuntos train y test
    X_train, X_test, y_train, y_test = train_test_split(x_preprocessed, y, test_size=0.2)
    #aplicar cross validation para hallar mejor valor del parametro
    bestalpha, bestmse = ridge_regressionCV(X_train, y_train, 10)
    #ajustar modelo
    model = Ridge(bestalpha).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #hallar error cuadratico medio
    mse = mean_squared_error(y_pred, y_test)
    #imprimir resultados
    print('Best alpha: ', bestalpha)
    print('Mean squared error: %.2f' % mse)
    print('Eint: {:.5f}'.format(1- model.score(X_train, y_train)))
    print('Eout: {:.5f}'.format(1-model.score(X_test, y_test)))
    input('\n--- Tap enter to continue ---\n')
    #pintar predicciones
    printError(y_test, y_pred)

##########################################################

print("Task 1. Classification problem\n")
classificationProblem('datos/digits/optdigits.tra', 'datos/digits/optdigits.tes')
print("Task 2. Regression problem\n")
ridge_regressionProblem('datos/airfoil/airfoil_self_noise.dat')
input('\n--- Tap enter to end ---\n')
