"""
@author: Horax & Fedex
"""
#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#%%
"""Definimos los polinomios y coeficientes que vamos a usar"""

promedio_cross_x = [-0.03461811951503092,0] #La pendiente y una o.o en 0 para la función y = mx
promedio_cross_y = [-0.07531236878047691,0] #La pendiente y una o.o en 0 para la función x = my
m1 = promedio_cross_x[0]
m2 = promedio_cross_y[0]

datos_ajuste_x = pd.read_csv(r'ajuste_cubico_x_calv2.csv') 
datos_ajuste_y= pd.read_csv(r'ajuste_cubico_y_calv2.csv')

promedio_x = datos_ajuste_x.mean()[1::] # Los datos de los ajustes para todas las filas de las no linealidades en la ida
coefs_x = [coef for coef in promedio_x] # Coeficientes de V_x(x) = ax^3 + bx^2 + cx
coefs_x[-1] = 0


promedio_y = datos_ajuste_y.mean()[1::] # Los datos de los ajustes para todas las filas de las no linealidades en la ida
coefs_y = [coef for coef in promedio_y] # Coeficientes de V_x(x) = ax^3 + bx^2 + cx
coefs_y[-1] = 0

def x_req(x_trg,y_trg):
    return (x_trg - m2*y_trg)/(1-m1*m2)

def y_req(x_trg,y_trg):
    return (y_trg - m1*x_trg)/(1-m1*m2)

polinomio_x = np.poly1d(coefs_x)
polinomio_y = np.poly1d(coefs_y)

def crear_dcs(x_0,y_0,x_f,y_f,pasos_x,pasos_y): 
    x_trgs = np.linspace(x_0,x_f,pasos_x)
    y_trgs = np.linspace(y_0,y_f,pasos_y)
    dcsx_grid = []
    dcsy_grid = []
    
    for y in y_trgs:
        for x in x_trgs:
            # Aplicar matriz inversa de cross-talk (vectorizado por punto)
            xr = x_req(x, y)
            yr = y_req(x, y)
            
            # Evaluar polinomios de histéresis para obtener el Duty Cycle / Voltaje
            vx = polinomio_x(xr)
            vy = polinomio_y(yr)
            
            dcsx_grid.append(vx)
            dcsy_grid.append(vy)
            
    return np.array(dcsx_grid), np.array(dcsy_grid)

#%%
dcsx, dcsy = crear_dcs(0,12,0,12,20,20)
dutys_csv = pd.DataFrame({'Dcx': dcsx,'Dcy':dcsx})
dutys_csv.to_csv(r'dutys_v4_fede_csv.csv')