# -*- coding: utf-8 -*-
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import time

# 1. Configuración de rutas y parámetros
trayectorias_path = r'Analisis de video\Datos_tray\Discreto_xy_nocorrect_0805.csv'
data = pd.read_csv(trayectorias_path)
x_traj = data['X'].values  
y_traj = data['Y'].values

plt.plot(x_traj, y_traj, label='Trayectoria de la partícula')
plt.gca().invert_yaxis() # Invertimos Y porque en imágenes el (0,0) es la esquina superior
plt.show()