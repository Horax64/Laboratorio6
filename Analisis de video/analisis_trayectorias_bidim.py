# -*- coding: utf-8 -*-
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import time

# 1. Configuración de rutas y parámetros
trayectorias_path = r'Analisis de video\Datos_tray\Barrido_continuo_vertical_1505.csv'
data = pd.read_csv(trayectorias_path)

plt.scatter(data['X'], data['Y'], s = 10)
plt.show()

cross = []
y_0 = []
for tiempos in data['t_0_video'].unique():
    mask = data['t_0_video'] == tiempos
    x_traj = data[mask]['X'].values
    y_traj = data[mask]['Y'].values
    x_traj = x_traj[0:590]
    y_traj = y_traj[0:590]
    y_0.append(y_traj[0])
    coef, cov = np.polyfit(x_traj, y_traj, deg=1,cov=True)
    print(coef[0], cov[0][0])
    cross.append((coef[0],cov[0][0]))

plt.scatter(range(0,len(cross)), [c[0] for c in cross], label='Pendiente de la trayectoria')
plt.show()