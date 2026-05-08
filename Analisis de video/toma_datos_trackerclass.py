# -*- coding: utf-8 -*-
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
import time

# 1. Configuración de rutas y parámetros
file_name = 'Discreto_xy_nocorrect_0805'
video_path = rf'C:\Users\LEC\Desktop\Garcia Crespo-Arias Ceci\Análisis de vídeo\{file_name}.mp4'  # Cambia esto por el nombre de tu archivo
fps = tracker.fps(video_path)

tiempos = np.array(range(0, 419, 22)) # Tiempos en segundos donde se realizarán los trackeos
tiempos = tiempos + 0.5

x_traj = []
y_traj = []

df_trayectoria = pd.DataFrame({'X': x_traj, 'Y': y_traj, 't_0_video' : []})

for tiempo in tiempos:
    try:
        t_0 = tiempo # Tiempo inicial en segundos
        frame_inicial = int(fps*t_0)            # Frame donde quieres empezar
        t_f = tiempo + 20
        frame_final =  int(fps*t_f)

        ancho_busqueda = [50, 50]    # Tamaño del área roja (donde busca a la partícula)
        velocidad_visualizacion = 1  # ms entre frames (1 es lo más rápido)

        # 2. Selección interactiva del Template (la partícula)
        # Se abrirá una ventana, selecciona la partícula con el mouse y presiona ENTER o ESPACIO
        centro, ancho_template = tracker.setTemplate(video_path, frame_inicial,0)

        # 3. Inicialización de matrices
        template, obs = tracker.inicio(video_path, centro, ancho_template, ancho_busqueda, frame_inicial, 0)

        # 4. Ejecución del trackeo
        duracion = [frame_inicial, frame_final] 
        x_traj, y_traj = tracker.corr(video_path, template, obs, centro, velocidad_visualizacion, duracion, 0)
        tiempo_video = np.ones(len(x_traj))*tiempo  # Tiempo correspondiente a cada frame trackeado
        df_trayectoria = pd.concat([df_trayectoria, pd.DataFrame({'X': x_traj, 'Y': y_traj, 't_0_video': tiempo_video})], ignore_index=True)

    except:
        df_trayectoria.to_csv(f'{file_name}.csv', index=False)  # Guardar la trayectoria en un archivo CSV
        break
df_trayectoria.to_csv(f'{file_name}.csv', index=False)

# 5. Visualización de resultados con Matplotlib
plt.figure(figsize=(8, 6))
plt.plot(x_traj, y_traj, label='Trayectoria de la partícula')
plt.gca().invert_yaxis() # Invertimos Y porque en imágenes el (0,0) es la esquina superior
plt.xlabel("X (pixeles)")
plt.ylabel("Y (pixeles)")
plt.title("Trayectoria recuperada")
plt.legend()
plt.show()