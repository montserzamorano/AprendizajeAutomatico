# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Montserrat Rodriguez Zamorano
"""
import numpy as np
import random
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(13)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
def ejercicio11a():
    x = simula_unif(50, 2, [-50,50])
    plt.plot(x,'o',markersize=2)
    plt.title("Nube de puntos")
    plt.xlabel("eje x")
    plt.ylabel("eje y") 
    plt.show()  
    input("\n--- Pulsar enter para continuar ---\n")
def ejercicio11b():
    x = simula_gaus(50, 2, np.array([5,7]))
    plt.plot(x,'o',markersize=2)
    plt.title("Nube de puntos 2")
    plt.xlabel("eje x")
    plt.ylabel("eje y")
    plt.show()  
    input("\n--- Pulsar enter para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def f1(x):
    return (x[:,0]-10)**2 + (x[:,1]-20)**2 -400

def f2(x):
    return 0.5*((x[:,0]+10)**2)+((x[:,1]-20)**2)-400

def f3(x):
    return 0.5*((x[:,0]-10)**2)-((x[:,1]-20)**2)-400

def f4(x):
    return x[:,1]-20*x[:,0]**2-5*x[:,0]+3
###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()

#generamos muestra de puntos 2D
u = simula_unif(50, 2, [-50,50])
#generamos recta para añadir etiqueta
r = simula_recta([-50,50])
x_arr = np.array([-50,50])
y = r[0]*x_arr+r[1]
num_pos = 0
num_neg = 0

def generaEtiqueta(datos, recta):
    et = []
    np = 0
    nn = 0
    for i in range(len(datos)):
       if (f(datos[i,0],datos[i,1],recta[0],recta[1]) == 1) :
           np += 1
           et.append(1)
       else:
           nn += 1
           et.append(-1)
    return et, np, nn

etiq, num_pos, num_neg = generaEtiqueta(u,r)
etiq = np.array(etiq)
  
def generarRuido(e, porc, np, nn):
    #modificar de forma aleatoria etiquetas
    num_mod_pos = int(porc*np)
    num_mod_neg = int(porc*nn)
    #print("Número de etiquetas positivas a modificar: ", num_mod_pos, "\n")
    #print("Número de etiquetas negativas a modificar: ", num_mod_neg, "\n")
    it_pos = 0
    it_neg = 0
    for i in range(np+nn):
        if e[i]==1:
            if it_pos<num_mod_pos:
                e[i]=-1
                it_pos += 1
                #print("Modificada etiqueta positiva\n")
        elif e[i]==-1:
            if it_neg<num_mod_neg:
                e[i]=1
                it_neg += 1
                #print("Modificada etiqueta negativa\n")
    return e

def ejercicio1_23():
    e = np.copy(etiq)
    plt.scatter(u[:,0],u[:,1], c=e, s=3)
    plt.plot(x_arr, r[1]+r[0]*x_arr)
    #dibujamos recta
    plt.title("Clasificación usando signo")
    plt.xlabel("eje x")
    plt.ylabel("eje y")
    plt.show()
    input("\n--- Pulsar enter para continuar ---\n")
    e = generarRuido(e, 0.1, num_pos, num_neg)
     
    plt.scatter(u[:,0],u[:,1], c=e, s=3)
    plt.plot(x_arr, r[1]+r[0]*x_arr)
    #dibujamos recta
    plt.title("Clasificación usando signo con ruido")
    plt.xlabel("eje x")
    plt.ylabel("eje y")
    plt.show()
    input("\n--- Pulsar enter para continuar ---\n")
    #usamos funciones mas complejas como frontera de clasificacion
    plot_datos_cuad(u,e,f1,'Clasificación con ruido usando f1','eje x','eje y')
    input("\n--- Pulsar enter para continuar ---\n")
    plot_datos_cuad(u,e,f2,'Clasificación con ruido usando f2','eje x','eje y')
    input("\n--- Pulsar enter para continuar ---\n")
    plot_datos_cuad(u,e,f3,'Clasificación con ruido usando f3','eje x','eje y')
    input("\n--- Pulsar enter para continuar ---\n")
    plot_datos_cuad(u,e,f4,'Clasificación con ruido usando f4','eje x','eje y')
    input("\n--- Pulsar enter para continuar ---\n")



###############################################################################
###############################################################################
###############################################################################

#convierte los pesos en nuestra linea solucion
def getLine(w,x):
    a = -w[1]/w[2]
    b = -w[0]/w[2]
    return a*x+b

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

# datos: matriz donde cada item con su etiqueta está representado
# por una fila de la matriz
# label: vector de etiquetas
# max_iter: número máximo de etiquetas permitidas
# vini: valor inicial del vector

def ajusta_PLA(datos, label, max_iter, vini):
    it = 0
    error = 10**(-12)
    w_old = vini
    w_new = vini
    while it < max_iter:
        it +=1
        w_old = w_new
        for i in range(len(datos)):
            if signo(np.dot(np.transpose(w_new),datos[i]))!=label[i]:
                w_new = w_old + label[i]*datos[i]
                #print(w_new)
        #si converge, parar la iteracion
        if(np.linalg.norm(w_new-w_old)<error):
            break
    return w_old, it  

def inicializarDatos(x1):
    xtemp = []
    for i in range(len(x1)):
        xtemp.append([1, x1[i][0],x1[i][1]])
    return np.array(xtemp)

def ejercicio2_1a():
    # Random initializations
    iterations = []
    x_ = inicializarDatos(u)
    #e_ = generarRuido(etiq, 0.1, num_pos, num_neg)
    w, it = ajusta_PLA(x_, etiq, 10000, [0,0,0])
    iterations.append(it)
    print("Salida ", w)
    print("Número iteraciones ",it)
    plt.scatter(u[:,0],u[:,1], c=etiq, s=3)
    plt.plot(x_arr, getLine(w, x_arr))
    plt.plot(x_arr, r[1]+r[0]*x_arr, c='y')
    plt.title("PLA con w inicial vector cero")
    plt.xlabel("eje x")
    plt.ylabel("eje y")
    plt.show()
    input("\n--- Pulsar enter para continuar ---\n")
    print("\nw inicial vector aleatorio\n")
    for i in range(0,10):
        wini = np.random.rand(3)
        w, it = ajusta_PLA(inicializarDatos(u), etiq, 10000, wini)
        iterations.append(it)
        print("Vector inicial:", wini)
        print("Salida ", w)
        print("Número iteraciones ",it)
        plt.title("PLA con w inicial vector aleatorio")
        plt.xlabel("eje x")
        plt.ylabel("eje y")
        plt.scatter(u[:,0],u[:,1], c=etiq, s=3)
        plt.plot(x_arr, getLine(w,x_arr))
        plt.plot(x_arr, r[1]+r[0]*x_arr, c='y')
        plt.show()
        input("\n--- Pulsar enter para continuar ---\n")
    print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))



def ejercicio2_1b():
    # Random initializations
    iterations = []
    x_ = inicializarDatos(u)
    e_ = np.copy(etiq)
    e_ = generarRuido(e_, 0.1, num_pos, num_neg)
    w, it = ajusta_PLA(x_, e_, 10000, [0,0,0])
    iterations.append(it)
    print("Salida ", w)
    print("Número iteraciones ",it)
    plt.scatter(u[:,0],u[:,1], c=e_, s=3)
    plt.plot(x_arr, getLine(w, x_arr))
    plt.plot(x_arr, r[1]+r[0]*x_arr, c='y')
    plt.title("PLA con w inicial vector cero")
    plt.xlabel("eje x")
    plt.ylabel("eje y")
    plt.show()
    input("\n--- Pulsar enter para continuar ---\n")
    print("\nw inicial vector aleatorio\n")
    for i in range(0,10):
        wini = np.random.rand(3)
        w, it = ajusta_PLA(inicializarDatos(u), e_, 10000, wini)
        iterations.append(it)
        print("Vector inicial:", wini)
        print("Salida ", w)
        print("Número iteraciones ",it)
        plt.title("PLA con w inicial vector aleatorio")
        plt.xlabel("eje x")
        plt.ylabel("eje y")
        plt.scatter(u[:,0],u[:,1], c=e_, s=3)
        plt.plot(x_arr, getLine(w, x_arr))
        plt.plot(x_arr, r[1]+r[0]*x_arr, c='y')
        plt.show()
        input("\n--- Pulsar enter para continuar ---\n")
    print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

#codigo funcion sigmoide
def sigmoide(x):
    return 1/(1+np.e**(-x))

def gradiente(y,w,x):
    temp0 = np.dot(-y,x)
    temp1 = sigmoide(np.dot(-y,np.dot(np.transpose(w),x)))
    return temp0*temp1

def porcentajeError(y,w,x):
    error = 0
    h_arr = []
    for i in range(len(x)):
        #hallamos la prediccion
        h = sigmoide(np.dot(np.transpose(w),x[i]))
        if h>=0.5:
            h = 1
        else:
            h = -1
        h_arr.append(h)
        if(h != y[i]):
            error+= 1
    return (error/len(x))*100
     
def sgdRL(datos, label, lr, max_iter, vini, error):
    it = 0
    w_ant = np.array(vini)
    w_new = np.array(vini)
    indices = np.arange(len(datos))

    while it < max_iter:
        it += 1
        #aplicamos una permutacion aleatoria en el orden de los datos
        np.random.shuffle(indices)
        datos = datos[indices]
        label = np.array(label)[indices]
        #pase de los N datos
        for i in range(len(datos)):
            gr = gradiente(label[i], w_new, datos[i])
            w_new = w_new - lr*gr
        #en caso de convergencia, detenemos la iteracion
        if np.linalg.norm(w_ant-w_new) < error:
            break
        #actualizamos
        w_ant = w_new
    return w_new, it

def ejercicio2_2a():
    u_ = simula_unif(100, 2, [0,2])
    r_ = simula_recta([0,2])
    x_arr2 = np.array([0,2])
    
    et, num_pos, num_neg = generaEtiqueta(u_,r_)
    
    x_ = inicializarDatos(u_)
    w,it = sgdRL(x_, et, 0.01, 100, [0, 0, 0], 0.01)
    
    plt.scatter(u_[:,0],u_[:,1], c=et, s=3)
    plt.plot(x_arr2, getLine(w, x_arr2))
    plt.plot(x_arr2, r_[0]*x_arr2+r_[1], c='y')
    plt.title("RL con SGD")
    plt.show()
    
    print("Porcentaje de error: ", porcentajeError(et,w,x_))
    print("Iteraciones: ", it)
    input("\n--- Pulsar enter para continuar ---\n")
 
# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).    

def ejercicio2_2b():
    u_ = simula_unif(1000, 2, [0,2])
    r_ = simula_recta([0,2])
    x_arr2 = np.array([0,2])
    
    et, num_pos, num_neg = generaEtiqueta(u_,r_)
    
    x_ = inicializarDatos(u_)
    w,it = sgdRL(x_, et, 0.01, 100, [0, 0, 0], 0.01)
    
    plt.scatter(u_[:,0],u_[:,1], c=et, s=3)
    plt.plot(x_arr2, getLine(w, x_arr2))
    plt.plot(x_arr2, r_[0]*x_arr2+r_[1], c='y')
    plt.title("RL con SGD")
    plt.show()
    
    print("Porcentaje de error: ", porcentajeError(et,w,x_))
    print("Iteraciones: ", it)
    input("\n--- Pulsar enter para continuar ---\n")    
    


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
'''def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

'''

######### LLAMADA A LOS EJERCICIOS ####################3

print("APRENDIZAJE AUTOMATICO. PRACTICA 2\n")
print("COMPLEJIDAD DE H Y EL RUIDO \n")
print("Ejercicio 1. Apartado a\n")
ejercicio11a()
print("Ejercicio 1. Apartado b\n")
ejercicio11b()
print("Ejercicios 2 y 3.\n")
ejercicio1_23()
print("MODELOS LINEALES \n")
print("Ejercicio 1.\n")
ejercicio2_1a()
ejercicio2_1b()
print("Ejercicio 2.\n")
ejercicio2_2a()
ejercicio2_2b()
print("FIN DE LA PRÁCTICA")