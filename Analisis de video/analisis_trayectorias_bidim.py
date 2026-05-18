# -*- coding: utf-8 -*-
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import time

# 1. Configuración de rutas y parámetros
trayectorias_path = r'Analisis de video\Datos_tray\Discreto_x_1805.csv'
data = pd.read_csv(trayectorias_path)

plt.scatter(data['X'], data['Y'], s = 10)
plt.show()

def distancia_euclidiana(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

saltos=[0]

for i in range(1, len(data)-1):
    p_anterior = (data['X'][i-1], data['Y'][i-1])
    p_actual = (data['X'][i], data['Y'][i])
    p_siguiente = (data['X'][i+1], data['Y'][i+1])
    
    d_anterior = distancia_euclidiana(p_anterior, p_actual)
    d_siguiente = distancia_euclidiana(p_actual, p_siguiente)
    
    if d_anterior < 20 and d_siguiente > 20:  # Umbral de distancia para detectar saltos
        saltos.append(i)
        
print(saltos)

for i in range(len(saltos)-1):
    plt.scatter(data['X'][saltos[i]:saltos[i+1]], data['Y'][saltos[i]:saltos[i+1]], s = 50, color='red')
    plt.show()