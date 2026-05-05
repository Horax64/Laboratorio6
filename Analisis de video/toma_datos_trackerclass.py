# -*- coding: utf-8 -*-
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import time

# 1. Configuración de rutas y parámetros
video_path = r'C:\Users\LEC\Desktop\Garcia Crespo-Arias Ceci\Análisis de vídeo\Discreto_ida_vuelta_x_2804.mp4'  # Cambia esto por el nombre de tu archivo
fps = tracker.fps(video_path)
t_0 = 12 # Tiempo inicial en segundos
frame_inicial = int(fps*t_0)            # Frame donde quieres empezar
t_f = 220
frame_final =  int(fps*t_f)

ancho_busqueda = [50, 50]    # Tamaño del área roja (donde busca a la partícula)
velocidad_visualizacion = 1  # ms entre frames (1 es lo más rápido)

# 2. Selección interactiva del Template (la partícula)
# Se abrirá una ventana, selecciona la partícula con el mouse y presiona ENTER o ESPACIO
centro, ancho_template = tracker.setTemplate(video_path, frame_inicial)

# 3. Inicialización de matrices
template, obs = tracker.inicio(video_path, centro, ancho_template, ancho_busqueda, frame_inicial)

# 4. Ejecución del trackeo
duracion = [frame_inicial, frame_final] 
x_traj, y_traj = tracker.corr(video_path, template, obs, centro, velocidad_visualizacion, duracion, 0)

df_trayectoria = pd.DataFrame({'X': x_traj, 'Y': y_traj})
df_trayectoria.to_csv('prueba.csv', index=False)  # Guardar la trayectoria en un archivo CSV


# 5. Visualización de resultados con Matplotlib
plt.figure(figsize=(8, 6))
plt.plot(x_traj, y_traj, label='Trayectoria de la partícula')
plt.gca().invert_yaxis() # Invertimos Y porque en imágenes el (0,0) es la esquina superior
plt.xlabel("X (pixeles)")
plt.ylabel("Y (pixeles)")
plt.title("Trayectoria recuperada")
plt.legend()
plt.show()