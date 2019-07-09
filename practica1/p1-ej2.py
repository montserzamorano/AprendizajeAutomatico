t# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Montserrat Rodríguez Zamorano
"""

import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from pylab import *
from sklearn import linear_model

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

#prediccion
def h(x,w):
    return np.dot(x,w)

# Funcion para calcular el coste/error
def Err(x,y,w):
    m = len(x)
    cost = np.sum(np.square(h(x,w)-y))/m
    return cost
    
def minibatch(x,y,tam):
    minibX = []
    minibY = []
    for i in range(tam):
        temp = np.random.choice(len(x))
        minibX.append(x[temp])
        minibY.append(y[temp])
    return minibX,minibY

def gradient(x,y,w):
    predminusy = h(np.transpose(x),w)-y
    return (np.dot(x,predminusy))

# Gradiente Descendente Estocastico
def sgd(x,y, learning_rate,maxIterations,error,m):
    #inicializar
    iterations = 0
    w = 0
    
    while iterations < maxIterations:
        cost = 0
        #generamos secuencia de minibatches
        minibatchX, minibatchY = minibatch(x,y,m)
        #iteramos en el minibatch
        for i in range(m):
            w_ant = w            
            #actualizamos los pesos
            w = w_ant - (2/m)*learning_rate*gradient(minibatchX,minibatchY,w_ant)
        #si el error es minimo, detenemos la iteracion
        if Err(minibatchX,minibatchY,w)<error:
            break
        iterations += 1
    return w, iterations, cost

# Pseudoinversa	
def pseudoinverse(x,y):
    t = np.transpose(x)
    pseudoinv = np.dot(np.linalg.inv(np.dot(t,x)),t)
    w = np.dot(pseudoinv,y)
    return w

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

w = sgd(x,y,0.01,50,1e-14,20)
print ('Bondad del resultado para grad. descendente estocastico:\n')
#print ("Eout: ", Err(x_test, y_test, w))
#print ("Ein: ", Err(x,y,w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
'''def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))'''

print("TRABAJO 1. APRENDIZAJE AUTOMATICO.\n")
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')