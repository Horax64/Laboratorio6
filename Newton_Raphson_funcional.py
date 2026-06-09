"""
@author: Horax & Fedex
"""

import pandas as pd 
import numpy as np


promedio_cross_x = [-0.03461811951503092,0]
promedio_cross_y = [-0.07531236878047691,0]

datos_ajuste_x = pd.read_csv('\ajuste_cubico_x.csv') 
datos_ajuste_y= pd.read_csv('\ajuste_cubico_y.csv')

promedio_x = datos_ajuste_x.mean()[1::]
coefs_x = [coef for coef in promedio_x] 
coefs_x[-1] = 0


promedio_y = datos_ajuste_y.mean()[1::]
coefs_y = [coef for coef in promedio_y] 
coefs_y[-1] = 0


cross_x = np.poly1d(promedio_cross_x)
cross_y = np.poly1d(promedio_cross_x)


polinomio_x = np.poly1d(coefs_x)
polinomio_y = np.poly1d(coefs_y)

print(polinomio_x)

d_polinomio_x = np.polyder(polinomio_x)
d_polinomio_y = np.polyder(polinomio_y)



# Newton-Raphson para un paso

def NR_1paso_x(dc_x0,X):
    F_x = polinomio_x(dc_x0) - X
    return dc_x0 - F_x/d_polinomio_x(dc_x0)


def NR_1paso_y(dc_y0,X,Y):
    F_y = polinomio_y(dc_y0) + cross_x(X)- Y
    return dc_y0 - F_y/d_polinomio_y(dc_y0)

# Newton-Raphson en n iteraciones

def NR_npasos_x(dc_x0,X,n):
    dc_x = NR_1paso_x(dc_x0,X)
    for _ in range(n-1):
         dc_x = NR_1paso_x(dc_x,X)
    return dc_x

def NR_npasos_y(dc_y0,X,Y,n):
    dc_y = NR_1paso_y(dc_y0,X,Y)
    for _ in range(n-1):
         dc_y = NR_1paso_y(dc_y,X,Y)
    return dc_y

def dc_scanning(paso_x,paso_y):

    max_duty = 1

    cant_pasos_x = int(max_duty/paso_x)
    cant_pasos_y = int(max_duty/paso_y)

    dc_x_inicial = np.linspace(0,max_duty,cant_pasos_x)
    dc_y_inicial = np.linspace(0,max_duty,cant_pasos_y)

    X_final = np.linspace(0,polinomio_x(1),cant_pasos_x)
    Y_final = np.linspace(0,polinomio_y(1),cant_pasos_y)
    dcs_x = []
    dcs_y = []

    for dc_x0,x in zip(dc_x_inicial,X_final):
        dcx = NR_npasos_x(dc_x0,x,20)
        dcs_x.append(dcx)
        
        for dc_y, y in zip(dc_y_inicial,Y_final):
            dcy = NR_npasos_y(dc_y,x,y,20)
            dcs_y.append(dcy)

    return dcs_x, dcs_y

