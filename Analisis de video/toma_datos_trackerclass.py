# -*- coding: utf-8 -*-
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time

# 1. Configuración de rutas y parámetros
video_path = r'C:\Users\LEC\Desktop\Garcia Crespo-Arias Ceci\Análisis de vídeo\Barrido_ida_vuelta_x.mp4'  # Cambia esto por el nombre de tu archivo
frame_inicial = 200            # Frame donde quieres empezar
ancho_busqueda = [50, 50]    # Tamaño del área roja (donde busca a la partícula)
velocidad_visualizacion = 1  # ms entre frames (1 es lo más rápido)

# 2. Selección interactiva del Template (la partícula)
# Se abrirá una ventana, selecciona la partícula con el mouse y presiona ENTER o ESPACIO
centro, ancho_template = tracker.setTemplate(video_path, frame_inicial)

# 3. Inicialización de matrices
template, obs = tracker.inicio(video_path, centro, ancho_template, ancho_busqueda, frame_inicial)

# 4. Ejecución del trackeo
# El segundo parámetro de 'duracion' es el frame final.
duracion = [frame_inicial, 3000] 
x_traj, _ = tracker.corr(video_path, template, obs, centro, velocidad_visualizacion, duracion)

# 5. Visualización de resultados con Matplotlib
plt.figure(figsize=(8, 6))
plt.plot(range(frame_inicial,5602+frame_inicial), x_traj, label='Trayectoria de la partícula')
plt.gca().invert_yaxis() # Invertimos Y porque en imágenes el (0,0) es la esquina superior
plt.xlabel("X (píxeles)")
plt.ylabel("Y (píxeles)")
plt.title("Trayectoria recuperada")
plt.legend()
plt.show()