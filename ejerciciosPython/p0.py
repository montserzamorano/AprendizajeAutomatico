#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:36:19 2019

@author:Montse Rodríguez
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#Datos acerca del iris data set
# 4 atributos
# 150 instancias
# 3 clases, 50 instancias por clase

#Parte 1
#1. leer la base de datos de iris que hay en scikit-learn
iris = datasets.load_iris()
#2. obtener las caracteristicas(datos de entrada X) y la clase (y)
x = iris.data
y = iris.target
nombres = iris.target_names
#3. quedarse con las dos ultimas caracteristicas
carac = x[:, 2:4]
#4. visualizar con un scatter plot los datos, coloreando cada clase con un
#color diferente.
#carac[:,0][y==tipo] conjunto de los elementos del array que cumplen que los tipos son iguales
for tipo, color in zip([0,1,2], "rgb"):
    plt.scatter(carac[:,0][y==tipo], carac[:,1][y==tipo],c=color, label=nombres[tipo])
plt.title("Iris data set")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend(loc="upper left")
plt.show()


#Parte 2
#Separar en training y test aleatoriamente conservando la proporcion de 
#elementos en cada clase tanto en training como en test.
#separamos por tipos
datos0=x[y==0]
datos1=x[y==1]
datos2=x[y==2]
#hacemos shuffle para que el orden sea aleatorio
np.random.shuffle(datos0)
np.random.shuffle(datos1)
np.random.shuffle(datos2)
#usamos 40 por ser el 80% de 50 instancias por clase
#concatenamos
training = np.concatenate((datos0[:40,:],datos1[:40,:],datos2[:40,:]))
testing = np.concatenate((datos0[40:,:],datos1[40:,:],datos2[40:,:]))


#Parte 3
#1. Obtener 100 valores equiespaciados entre0 y 2pi.
n = 100
a = np.linspace(0,2*np.pi,n)
#2. Obtener el valor de sinx, cosx y sinx+cosx para los 100 valores
senos = np.sin(a)
cosenos = np.cos(a)
suma = senos+cosenos
#3. Visualizar las tres curvas simultaneamente en el mismo plot
plt.plot(a,senos, 'k--',label="sin(x)")
plt.plot(a,cosenos, 'b--', label="cos(x)")
plt.plot(a,suma, 'r--', label="sin(x)+cos(x)")
plt.legend() 
plt.show()   