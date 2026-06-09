"""
@author: Horax & Fedex
"""
#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#%%
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
cross_y = np.poly1d(promedio_cross_y) #Desplazamiento en y para un salto en x (px o micrones) (\Delta_x/\Delta_y)


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
    F_y = polinomio_y(dc_y0) + cross_x(X) - Y
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

paso = 0.05
cant_pasos = int(1/paso)
dcsx_0,dcsy_0 = dc_scanning(paso,paso)
#print(np.array(dcsy_0).reshape(20,20).T.reshape(1,400))

#%%
# Ordenamos los arrays
dcs_x = []
dcs_y = np.array(dcsy_0).reshape(cant_pasos,cant_pasos).T.reshape(1,cant_pasos**2)
#print(dcs_y[0])
dcs_y = dcs_y[0]

for _ in range(0,cant_pasos):
    dcs_x.extend(dcsx_0)

#Restringimos el dominio para evitar problemas de borde
borde_a_eliminar = 2
extremo = cant_pasos*borde_a_eliminar
dcs_x_recort = dcs_x[extremo:-extremo]
dcs_y_recort = dcs_y[extremo:-extremo]
to_drop = []

for i in range(0,len(dcs_y_recort)):
    n = i%cant_pasos
    restos = np.array(range(0,borde_a_eliminar,1))
    restos = np.append(restos, np.array(range(cant_pasos-borde_a_eliminar,cant_pasos,1)))
    if n in restos:
        to_drop.append(i)
dcs_x_recort =np.delete(dcs_x_recort, to_drop)
dcs_y_recort = np.delete(dcs_y_recort, to_drop)

#Vemos que conseguimos
plt.scatter(dcs_x,dcs_y, c='grey', s = 100)
plt.scatter(dcs_x_recort,dcs_y_recort, c='r', s=50)
plt.show()

#%%
"""Visualzar que esperamos que ocurra para las posiciones
hasta ahora"""

ancho = cant_pasos - borde_a_eliminar*2

dcs_x_recort_rsh = dcs_x_recort.reshape(ancho,ancho).T
dcs_y_recort_rsh = dcs_y_recort.reshape(ancho,ancho).T

pos_y_final = []
pos_x_final = []
for i in range(0,ancho):
    columna_y = dcs_y_recort_rsh[i]
    columna_x = dcs_x_recort_rsh[i]
    posiciones_y = polinomio_y(columna_y)
    posiciones_y = posiciones_y - np.max(posiciones_y)/2
    pos_y_final = np.append(pos_y_final,posiciones_y)
    posiciones_x = polinomio_x(columna_x) + cross_y(posiciones_y)
    pos_x_final = np.append(pos_x_final,posiciones_x)
    
plt.scatter(pos_x_final.T.reshape(1,ancho**2),pos_y_final.T.reshape(1,ancho**2))
plt.show()

#%%
"""Corregir cross en 'Y'"""

def NR_1paso_x_2ord(dc_x0,X,Y):
    F_x = polinomio_x(dc_x0) + cross_y(Y) - X
    return dc_x0 - F_x/d_polinomio_x(dc_x0)

# Newton-Raphson en n iteraciones
def NR_npasos_x_2ord(dc_x0,X,Y,n):
    dc_x = NR_1paso_x_2ord(dc_x0,X,Y)
    for _ in range(n-1):
         dc_x = NR_1paso_x_2ord(dc_x,X,Y)
    return dc_x

def dc_x_correct(paso_x,paso_y,dcs_x):

    max_duty = 1

    cant_pasos_x = int(max_duty/paso_x)
    cant_pasos_y = int(max_duty/paso_y)

    dc_x_inicial = dcs_x
    dcs_x_correct = []

    X_final = np.linspace(0,polinomio_x(1),cant_pasos_x)
    Y_final = np.linspace(0,polinomio_y(1),cant_pasos_y)

    for y in Y_final:
        for dc_x0,x in zip(dc_x_inicial,X_final):
            dcx = NR_npasos_x_2ord(dc_x0,x,y,20)
            dcs_x_correct.append(dcx)

    return dcs_x_correct

dcs_x_correct = dc_x_correct(paso,paso,dcs_x)

borde_a_eliminar = 2
extremo = cant_pasos*borde_a_eliminar
dcs_x_recort = dcs_x_correct[extremo:-extremo]
dcs_y_recort = dcs_y[extremo:-extremo]
to_drop = []

for i in range(0,len(dcs_y_recort)):
    n = i%cant_pasos
    restos = np.array(range(0,borde_a_eliminar,1))
    restos = np.append(restos, np.array(range(cant_pasos-borde_a_eliminar,cant_pasos,1)))
    if n in restos:
        to_drop.append(i)
dcs_x_recort =np.delete(dcs_x_recort, to_drop)
dcs_y_recort = np.delete(dcs_y_recort, to_drop)

#Vemos que conseguimos
plt.scatter(dcs_x_recort,dcs_y_recort, c='r', s=50)
plt.show()

#%%
"""Visualzar que esperamos que ocurra para las posiciones
hasta ahora"""

ancho = cant_pasos - borde_a_eliminar*2

dcs_x_recort_rsh = dcs_x_recort.reshape(ancho,ancho).T
dcs_y_recort_rsh = dcs_y_recort.reshape(ancho,ancho).T

pos_y_final = []
pos_x_final = []
for i in range(0,ancho):
    columna_y = dcs_y_recort_rsh[i]
    columna_x = dcs_x_recort_rsh[i]
    posiciones_y = polinomio_y(columna_y)
    posiciones_y = posiciones_y - np.max(posiciones_y)/2
    pos_y_final = np.append(pos_y_final,posiciones_y)
    posiciones_x = polinomio_x(columna_x) + cross_y(posiciones_y)
    pos_x_final = np.append(pos_x_final,posiciones_x)
    
plt.scatter(pos_x_final.T.reshape(1,ancho**2),pos_y_final.T.reshape(1,ancho**2))
plt.show()


# %%
"""Guardado final de arrays"""

dutys_csv = pd.DataFrame({'Dcx': dcs_x_recort,'Dcy':dcs_y_recort})
dutys_csv.to_csv(r'dutys_v3_csv.csv')

# %%
