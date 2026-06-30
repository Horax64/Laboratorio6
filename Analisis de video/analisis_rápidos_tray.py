# -*- coding: utf-8 -*-
#%%
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import time

# 1. Configuración de rutas y parámetros
file = 'Discreto_x_2905_correcciones_proc'
trayectorias_path = fr'Analisis de video\Datos_tray\{file}.csv'
data = pd.read_csv(trayectorias_path)

plt.scatter(data['X'], data['Y'], s = 10)
plt.show()

#%%

def distancia_euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

lineas_scanning = []

i=1
inicio = 0
while i < len(data)-1:
    p_anterior = (data['X'][i-1], data['Y'][i-1])
    p_actual = (data['X'][i], data['Y'][i])
    
    dist_y = np.abs(data['Y'][i]-data['Y'][i-1])

    d_2_anterior = distancia_euclidiana(p_anterior, p_actual)
    
    if d_2_anterior >= 14 or dist_y >= 4:  # Umbral de distancia para detectar saltos
        fin = i-1
        lineas_scanning.append([inicio,fin])
        i+=1
        while distancia_euclidiana(p_anterior, p_actual) > 5 and dist_y > 3:
            p_anterior = (data['X'][i-1], data['Y'][i-1])
            p_actual = (data['X'][i], data['Y'][i])
            i+=1

        inicio = i
    
    i+=1
    
# Al salir del while, verificamos si quedó un último cluster abierto sin registrar
if inicio < len(data):
    lineas_scanning.append([inicio, len(data) - 1])

print("Lineas de scanning detectadas:", lineas_scanning)

# 1. Inicializamos un array del tamaño exacto del DataFrame, lleno de -1 (saltos)
tiempos_corte = np.ones(len(data)) * -1

# 2. Iteramos sobre los rangos y "pintamos" el tiempo correspondiente
for id0, id1 in lineas_scanning:
    tiempo_0 = data['t_0_video'].iloc[id0]
    
    # Asignamos tiempo_0 desde id0 hasta id1 (usamos +1 porque el límite superior es exclusivo en Python)
    tiempos_corte[id0:id1+1] = tiempo_0

# 3. Lo agregamos directamente como columna al DataFrame
data['Tiempos_corte'] = tiempos_corte


file_name = r'Analisis de video\Datos_tray\Prueba_separación_automática'
#data.to_csv(f'{file_name}.csv', index=False)  

print(f'Largo de data: {len(data)}')
print(f'Largo de tiempos_corte: {len(data["Tiempos_corte"])}')

#%%
'''Visualización rápida parámetros de ajuste'''
import pandas as pd
import matplotlib.pyplot as plt

datos_ajuste_x = pd.read_csv(r'C:\Users\LEC\Desktop\Laboratorio6\ajuste_lin_x_calv1_3006.csv')
datos_ajuste_x = datos_ajuste_x[5:]

plt.scatter(datos_ajuste_x['Fila'],datos_ajuste_x['m'])
plt.show()
# %%
'''Visualización rápida parámetros de ajuste'''
import pandas as pd
import matplotlib.pyplot as plt

datos_ajuste_x = pd.read_csv(r'C:\Users\LEC\Desktop\Laboratorio6\ajuste_cubico_x_calv1_3006.csv')
datos_ajuste_x = datos_ajuste_x[5:]

plt.scatter(datos_ajuste_x['Fila'],datos_ajuste_x['c'])
plt.show()
# %%
