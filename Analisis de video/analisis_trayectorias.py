# -*- coding: utf-8 -*-
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import time

# 1. Configuración de rutas y parámetros
trayectorias_path = r'Analisis de video\Datos_tray\trayectoria_particula.csv'
data = pd.read_csv(trayectorias_path)
x_traj = data['X'].values  
y_traj = data['Y'].values

def segmentar_trayectoria(data, col_duty='X', threshold=1.1, min_length=10):
    """
    Segmenta la trayectoria en escalones estables.
    
    Args:
        data (pd.DataFrame): Datos con tiempo y Duty Cycle.
        col_duty (str): Nombre de la columna del Duty Cycle.
        threshold (float): Tolerancia para considerar que el valor no ha cambiado.
        min_length (int): Cantidad mínima de frames para considerar un escalón válido.
    """
    # 1. Calculamos la diferencia entre puntos sucesivos
    data['diff'] = data[col_duty].diff().abs()
    
    # 2. Identificamos dónde el cambio es casi nulo (escalones)
    data['is_stable'] = data['diff'] <= threshold
    
    # 3. Etiquetamos cada grupo de puntos estables consecutivos
    data['step_id'] = (data['is_stable'] != data['is_stable'].shift()).cumsum()
    data_stable = data[data['is_stable']].copy()
    
    # 4. Filtramos por duración mínima para limpiar ruido de transición
    segmentos = []
    for step_id, group in data_stable.groupby('step_id'):
        if len(group) >= min_length:
            segmentos.append({
                'X (px)': group[col_duty].mean(),
                'start_frame': group.index.min(),
                'end_frame': group.index.max(),
                'n_frames': len(group)
            })
            
    return pd.DataFrame(segmentos)

# Ejemplo de uso:
df_steps = segmentar_trayectoria(data)
print(df_steps.head())


# Visualización de los escalones detectados (crudos)
plt.figure(figsize=(8, 6))
plt.plot(range(len(x_traj)), x_traj, label='Trayectoria de la partícula')
plt.gca().invert_yaxis() # Invertimos Y porque en imágenes el (0,0) es la esquina superior
plt.xlabel("tiempo (frames)")
plt.ylabel("X (pixeles)")
plt.title("Trayectoria recuperada")
plt.legend()
plt.show()


# Trayecotria en función del dcycle (escalones)
plt.figure(figsize=(8, 6))
plt.scatter(range(len(df_steps))[0:21], df_steps['X (px)'][0:21], label='Trayectoria ida', color='blue')
plt.scatter(range(19,-1,-1), df_steps['X (px)'][21::], label='Trayectoria vuelta', color='red')
plt.gca().invert_yaxis() # Invertimos Y porque en imágenes el (0,0) es la esquina superior
plt.xlabel("paso")
plt.ylabel("X (px)")
plt.title("Curva de histéresis en el eje 'X'")
plt.legend()
plt.show()

