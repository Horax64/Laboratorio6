# -*- coding: utf-8 -*-
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv

# 1. Configuración de rutas y parámetros
file_name = 'Discreto_x_1805'
video_path = rf'C:\Users\LEC\Desktop\Garcia Crespo-Arias Ceci\Análisis de vídeo\{file_name}.mp4'
fps = tracker.fps(video_path)

# Extraemos el número total de frames del video para controlar el final real
cap_temp = cv.VideoCapture(video_path)
total_frames = int(cap_temp.get(cv.CAP_PROP_FRAME_COUNT))
cap_temp.release()
t_inicio = 0

# Parámetros del experimento
frame_actual = int(fps * t_inicio) 
ancho_busqueda = [50, 50]    
velocidad_visualizacion = 10  

df_trayectoria = pd.DataFrame()

print(f"Iniciando procesamiento continuo. Total de frames del video: {total_frames}")

while frame_actual < total_frames:
    try:
        print(f"\n=======================================================")
        print(f" Esperando ROI en Frame: {frame_actual} | Tiempo aproximado: {frame_actual/fps:.2f}s")
        print(f"=======================================================")
        
        # 2. Selección interactiva del Template (se abrirá automáticamente tras cada salto detectado)
        centro, ancho_template = tracker.setTemplate(video_path, frame_actual, 0)

        # 3. Inicialización de matrices
        template, obs = tracker.inicio(video_path, centro, ancho_template, ancho_busqueda, frame_actual, 0)

        # 4. Ejecución del trackeo dinámico
        duracion = [frame_actual, total_frames] 
        
        # Ajusta 'limite_salto' (píxeles) y 'max_predicciones' (frames perdidos) según la velocidad de tu setup
        x_traj, y_traj, ultimo_frame, salto_detectado = tracker.corr(
            video_path, template, obs, centro, velocidad_visualizacion, duracion, canal=0,
            limite_salto=20, max_predicciones=3
        )
        
        # Si se capturaron datos válidos en este tramo, los estructuramos
        if len(x_traj) > 0:
            frames_segmento = np.arange(frame_actual, frame_actual + len(x_traj))
            tiempos_segmento = frames_segmento / fps
            
            df_segmento = pd.DataFrame({
                'X': x_traj, 
                'Y': y_traj, 
                'Frame': frames_segmento,
                'Tiempo_seg': tiempos_segmento
            })
            
            df_trayectoria = pd.concat([df_trayectoria, df_segmento], ignore_index=True)
            # Guardado preventivo por tramos (si se rompe el programa, conservas todo lo anterior)
            df_trayectoria.to_csv(f'{file_name}.csv', index=False)  

        if not salto_detectado:
            print("\nSe ha alcanzado el final del video o el usuario canceló.")
            break
            
        # El puntero avanza exactamente a donde el tracker detectó que inició el salto
        frame_actual = ultimo_frame+10

    except Exception as e:
        print(f"\n[!] Interrupción en el procesamiento: {e}")
        break

# Finalizado el bucle, guardamos el dataframe global definitivo
if not df_trayectoria.empty:
    df_trayectoria.to_csv(f'{file_name}.csv', index=False)
    print(f"\nProceso finalizado con éxito. Datos guardados en {file_name}.csv")

    # 5. Visualización de resultados con Matplotlib (Muestra todo el barrido continuo concatenado)
    plt.figure(figsize=(9, 7))
    plt.plot(df_trayectoria['X'], df_trayectoria['Y'], label='Trayectoria global unificada', color='b', alpha=0.6)
    plt.gca().invert_yaxis() 
    plt.xlabel("X (píxeles)")
    plt.ylabel("Y (píxeles)")
    plt.title("Trayectoria completa recuperada (Multi-barrido)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
else:
    print("\nNo se registraron datos en la trayectoria.")