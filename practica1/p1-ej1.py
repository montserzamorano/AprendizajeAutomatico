#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Montserrat Rodríguez Zamorano
"""

import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from pylab import *
from sklearn import linear_model

np.random.seed(1)

def E(u,v):
    return ((u**2*np.e**v)-2*(v**2)*np.e**(-u))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 4*((u**3)*(np.e**(2*v))+(v**2)*(u**2)*(np.e**(v-u))-2*(v**2)*u*(np.e**(v-u))-2*(v**4)*(np.e**(-2*u)))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*((u**4)*(np.e**(2*v))-4*(u**2)*v*(np.e**(v-u))-2*(v**2)*(u**2)*(np.e**(v-u))+8*(v**3)*(np.e**(-2*u)))

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

#Argumentos que se le pasan a la funcion
#   -fun : funcion a minimizar
#   -learning_rate: tasa de aprendizaje
#   -maxIterations: numero maximo de iteraciones
#   -error: error en el que parar la ejecucion
#   -ini_point: punto inicial
#   -grad: gradiente de la funcion
#Salida
#   -w : punto en el que se alcanza el minimo
#   -val_arr : array de valores de la funcion
#   -iterations : iteraciones totales
def gradient_descent(fun, learning_rate,maxIterations,error,ini_point, grad):
    #inicializar numero de iteraciones
    iterations = 0
    #inicializar w y w_arr
    w = np.array(np.float64([ini_point[0],ini_point[1]]))
    val_arr = [np.float64(fun(ini_point[0],ini_point[1]))]
    while iterations<maxIterations:
        #actualizar numero de iteraciones
        iterations +=1
        #guardamos el ultimo w calculado
        w_ant = np.array([w[0],w[1]])
        #actualizar w
        w = w - learning_rate*grad(w_ant[0],w_ant[1])
        #actualizar w_arr
        val_arr.insert(len(val_arr), np.float64(fun(w[0],w[1])))
        #si el error es minimo, detenemos la ejecucion
        if(abs(np.float64(fun(w[0],w[1])-fun(w_ant[0],w_ant[1])))<error):
            break
    return w, val_arr, iterations 

def ejercicio12a():
    print('Apartado (a)\nMostrar el gradiente:\n', E(symbols('u'), symbols('v')))
    input("\n--- Pulsar enter para continuar ---\n")
    
def ejercicio12b():   
    eta = 0.01 
    maxIter = 10000000000
    error2get = 1e-14
    initial_point = np.array([1.0,1.0],dtype=np.float64)
    w, w_arr, it = gradient_descent(E,eta,maxIter,error2get,initial_point, gradE)
    print('\nApartado (b) Calcular número de iteraciones.\n')
    print ('Numero de iteraciones: ', it)
    input("\n--- Pulsar enter para continuar ---\n")
    
def ejercicio12c():   
    print('\nApartado (c) Calcular coordenadas.\n')
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')    
    input("\n--- Pulsar enter para continuar ---\n")

def F(x,y):
    return (x**2)+2*(y**2)+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
def dFx(x,y):
    return 2*(2*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)+x)
def dFy(x,y):
    return 4*(np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)+y)
#Gradiente de F
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])     

def ejercicio13a():
    print('\nApartado (a) Generar grafico de como desciende el valor de la funcion con las iteraciones.\n')
    
    w, w_arr, it = gradient_descent(F, 0.01,50,1e-14,np.array([0.1,0.1]),gradF)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')') 
    plt.plot(w_arr)
    plt.title("Experimento con tasa de aprendizaje eta = 0.01")
    plt.xlabel("Numero de iteraciones")
    plt.ylabel("f(x,y)") 
    plt.show()  
    input("\n--- Pulsar enter para continuar ---\n")
    
    w, w_arr, it = gradient_descent(F, 0.1,50,1e-14,np.array([0.1,0.1]),gradF)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    plt.plot(w_arr)
    plt.title("Experimento con tasa de aprendizaje eta = 0.1")
    plt.xlabel("Numero de iteraciones")
    plt.ylabel("F(x,y)") 
    plt.show()  
    input("\n--- Pulsar enter para continuar ---\n")


def ejercicio13b():
    print("Ejercicio 3b")
    print("Valor inicial (0.1,0.1)")
    w, w_arr, it = gradient_descent(F,0.01,50,1e-14,([0.1,0.1]),gradF)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('Valor minimo: ', F(w[0],w[1]))
    print("Valor inicial (1,1)")
    w, w_arr, it = gradient_descent(F,0.01,50,1e-14,([1,1]),gradF)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')') 
    print ('Valor minimo: ', F(w[0],w[1]))
    print("Valor inicial (-0.5,-0.5)")
    w, w_arr, it = gradient_descent(F,0.01,50,1e-14,([-0.5,-0.5]),gradF)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('Valor minimo: ', F(w[0],w[1]))
    print("Valor inicial (-1,-1)")
    w, w_arr, it = gradient_descent(F,0.01,50,1e-14,([-1,-1]),gradF)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')') 
    print ('Valor minimo: ', F(w[0],w[1]))
    input("\n--- Pulsar enter para finalizar el ejercicio ---\n")
    
print("TRABAJO 1. APRENDIZAJE AUTOMATICO.\n")
print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('EJERCICIO 1.2\n')
ejercicio12a()
ejercicio12b()
ejercicio12c()
print('EJERCICIO 1.3\n')
ejercicio13a()
ejercicio13b()