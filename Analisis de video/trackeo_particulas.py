# Importación de librerías
from trackerclass_v4 import tracker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
from tkinter import filedialog
from pathlib import Path

ruta_archivo = Path(filedialog.askopenfilename(
    title="Abrir video a trackear",
    defaultextension=".mp4",
    initialdir=Path.home(),        
    filetypes=[("MP4","*.mp4*"),("All files", "*.*")]
))

if not ruta_archivo:
    print("No se seleccionó ningún archivo. Saliendo.")
    exit()

video = ruta_archivo.name 
nombre_video = ruta_archivo.stem   
ruta_video = ruta_archivo.parent        


# 1. Configuración de rutas y parámetros

video_path = rf'{ruta_video}\{video}'
fps = tracker.fps(video_path)

# Extraemos el número total de frames del video para controlar el final real
cap_temp = cv.VideoCapture(video_path)
total_frames = int(cap_temp.get(cv.CAP_PROP_FRAME_COUNT))
cap_temp.release()
t_inicio = 0

# Parámetros del trackeo
frame_actual = int(fps * t_inicio) 
ancho_busqueda = [50, 50]    
velocidad_visualizacion = 1  

# Poné esto en False para desactivar la ventana de visualización del tracking
# y acelerar mucho el procesamiento (no dibuja ni espera frames en pantalla).
MOSTRAR_VIDEO = True

# --- PARÁMETROS DE AUTOMATIZACIÓN DEL SALTO DE FILA/COLUMNA ---
# Elegí el modo de escaneo:
#   "filas"    -> al terminar una línea, X vuelve a un valor fijo y Y avanza un paso fijo.
#   "columnas" -> al terminar una columna, Y vuelve a un valor fijo y X avanza un paso fijo.
#                 (sirve igual si dentro de la columna te movés hacia arriba en Y: el
#                 tracking de corr() sigue la partícula sea cual sea la dirección; esto
#                 solo define cómo se predice el salto entre columnas)
MODO_ESCANEO = "filas"

# Según el modo, definimos qué coordenada es la "fija" (se resetea a un valor
# constante en cada salto) y cuál es la "incremental" (avanza un paso fijo
# respecto a la última posición). 0 = X, 1 = Y.
if MODO_ESCANEO == "filas":
    IDX_FIJO, IDX_INCREMENTO = 0, 1        # fijo = X, incremental = Y
elif MODO_ESCANEO == "columnas":
    IDX_FIJO, IDX_INCREMENTO = 1, 0        # fijo = Y, incremental = X
else:
    raise ValueError("MODO_ESCANEO debe ser 'filas' o 'columnas'")

# Estos dos valores NO se ingresan a mano: el script los calcula solo a partir
# de las 2 primeras selecciones manuales de ROI (línea 1 y línea 2). Se dejan
# en None acá porque todavía no se conocen.
VALOR_FIJO = None       # se calcula tras la 2da selección manual
PASO_INCREMENTO = None  # se calcula tras la 2da selección manual

# Cuántos frames esperar tras el salto detectado antes de intentar relocalizar
# (le da tiempo a la partícula a asentarse en la nueva línea).
FRAMES_ESPERA_SALTO = 10

# Cuántas líneas/saltos seguidos se toleran sin encontrar la partícula antes
# de rendirse y pedir ayuda manual como último recurso.
MAX_RELOCALIZACIONES_FALLIDAS = 3

# Inicializamos dataframe para los datos obtenidos del trackeo
df_trayectoria = pd.DataFrame()

print(f"Iniciando procesamiento continuo. Total de frames del video: {total_frames}")

# 2. Selección manual SOLO para la partícula inicial (una única vez)
print(f"\n=======================================================")
print(f" Selección inicial en Frame: {frame_actual} | Tiempo aproximado: {frame_actual/fps:.2f}s")
print(f"=======================================================")
centro, ancho_template = tracker.setTemplate(video_path, frame_actual, 0)
template, obs = tracker.inicio(video_path, centro, ancho_template, ancho_busqueda, frame_actual, 0)

fallos_relocalizacion = 0

while frame_actual < total_frames:
    try:
        # 3. Ejecución del trackeo dinámico sobre la línea actual
        duracion = [frame_actual, total_frames]

        # Ajusta 'limite_salto' (píxeles) y 'max_predicciones' (frames perdidos) según la velocidad de tu setup
        x_traj, y_traj, ultimo_frame, salto_detectado = tracker.corr(
            video_path, template, obs, centro, velocidad_visualizacion, duracion, canal=0,
            limite_salto=20, max_predicciones=3, mostrar_video=MOSTRAR_VIDEO
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
            df_trayectoria.to_csv(f'{nombre_video}.csv', index=False)

        if not salto_detectado:
            print("\nSe ha alcanzado el final del video o el usuario canceló.")
            break

        # --- RELOCALIZACIÓN AUTOMÁTICA (sin selección manual) ---
        if len(x_traj) > 0:
            ultima_pos = [x_traj[-1], y_traj[-1]]
        else:
            ultima_pos = centro  # no hubo datos nuevos en este tramo, usamos el último centro conocido

        frame_reloc = ultimo_frame + FRAMES_ESPERA_SALTO

        # --- CALIBRACIÓN AUTOMÁTICA (solo ocurre una vez, en el 1er salto) ---
        # Todavía no conocemos el patrón de salto: pedimos la 2da selección
        # manual de ROI (línea 2) y, comparándola con el final de la línea 1,
        # calculamos VALOR_FIJO y PASO_INCREMENTO. De ahí en más, todo automático.
        if VALOR_FIJO is None:
            print(f"\n=======================================================")
            print(f" Calibración: seleccioná la partícula al inicio de la 2da "
                  f"{'fila' if MODO_ESCANEO == 'filas' else 'columna'} "
                  f"(Frame: {frame_reloc})")
            print(f"=======================================================")
            centro, ancho_template = tracker.setTemplate(video_path, frame_reloc, 0)
            template, obs = tracker.inicio(video_path, centro, ancho_template, ancho_busqueda, frame_reloc, 0)

            VALOR_FIJO = centro[IDX_FIJO]
            PASO_INCREMENTO = centro[IDX_INCREMENTO] - ultima_pos[IDX_INCREMENTO]
            print(f"[OK] Patrón de salto calibrado (modo '{MODO_ESCANEO}') -> "
                  f"VALOR_FIJO={VALOR_FIJO} px, PASO_INCREMENTO={PASO_INCREMENTO} px. "
                  f"De acá en más la relocalización será automática.")

            frame_actual = frame_reloc
            fallos_relocalizacion = 0
            continue

        # Predicción según la geometría del escaneo: la coordenada fija vuelve
        # al valor calibrado, la incremental avanza el paso fijo desde la última línea.
        posicion_predicha = [0, 0]
        posicion_predicha[IDX_FIJO] = VALOR_FIJO
        posicion_predicha[IDX_INCREMENTO] = ultima_pos[IDX_INCREMENTO] + PASO_INCREMENTO

        print(f"\n[i] Intentando relocalización automática en frame {frame_reloc} "
              f"(última pos: {ultima_pos}, predicción: {posicion_predicha})...")

        x_nuevo, y_nuevo, exito = tracker.relocalizar(
            video_path, template, ancho_busqueda, posicion_predicha,
            frame_reloc, canal=0
        )

        if exito:
            print(f"[OK] Partícula relocalizada en ({x_nuevo}, {y_nuevo}).")
            centro = [x_nuevo, y_nuevo]
            # Refrescamos template/obs desde la nueva posición para que el
            # tracking de la siguiente línea arranque con una plantilla actualizada.
            template, obs = tracker.inicio(video_path, centro, ancho_template, ancho_busqueda, frame_reloc, 0)
            frame_actual = frame_reloc
            fallos_relocalizacion = 0
        else:
            fallos_relocalizacion += 1
            print(f"[!] No se pudo relocalizar automáticamente (intento {fallos_relocalizacion}/"
                  f"{MAX_RELOCALIZACIONES_FALLIDAS}).")

            if fallos_relocalizacion >= MAX_RELOCALIZACIONES_FALLIDAS:
                print("[!] Demasiados fallos de relocalización automática seguidos. "
                      "Se pide selección manual como último recurso.")
                centro, ancho_template = tracker.setTemplate(video_path, frame_reloc, 0)
                template, obs = tracker.inicio(video_path, centro, ancho_template, ancho_busqueda, frame_reloc, 0)
                frame_actual = frame_reloc
                fallos_relocalizacion = 0
            else:
                # Reintenta un poco más adelante en el video por si la partícula
                # todavía no llegó a la posición esperada.
                frame_actual = frame_reloc + FRAMES_ESPERA_SALTO

    except Exception as e:
        print(f"\n[!] Interrupción en el procesamiento: {e}")
        break

# Finalizado el bucle, guardamos el dataframe global definitivo en el .csv
if not df_trayectoria.empty:
    df_trayectoria.to_csv(rf'Analisis de video\Datos_tray\Datos_crudos\{nombre_video}.csv', index=False)
    print(f"\nProceso finalizado con éxito. Datos guardados en {nombre_video}.csv")
else:
    print("\nNo se registraron datos en la trayectoria.")

# Leemos los datos obtenidos y los graficamos
df_trayectoria = pd.read_csv(rf'Analisis de video\Datos_tray\Datos_crudos\{nombre_video}.csv')

#Visualización de resultados con Matplotlib (Muestra todo el barrido)
plt.figure(figsize=(9, 7))
plt.scatter(df_trayectoria['X'], df_trayectoria['Y'], label='Trayectoria global unificada', color='b', alpha=0.6)
plt.gca().invert_yaxis() 
plt.xlabel("X (píxeles)")
plt.ylabel("Y (píxeles)")
plt.title("Trayectoria del barrido")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()