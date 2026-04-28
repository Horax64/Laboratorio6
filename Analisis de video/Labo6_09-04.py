"""
@author: Tomás Obregón
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from scipy.signal import fftconvolve
import cv2
import os

font_path = '/Users/tomas/ev/tw-cen-mt.ttf'  # Ruta completa al archivo de fuente
font_prop = font_manager.FontProperties(fname=font_path)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def show_frames(frames, fps):
    # Loop para mostrar cada frame con un pequeño retraso entre cada uno
    for frame in frames:
        # Mostrar el frame
        cv2.imshow('Video', frame)

        # Esperar un breve tiempo (aproximadamente 1/FPS segundos)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break  # Salir si se presiona 'q'

    # Cerrar la ventana al finalizar la reproducción
    cv2.destroyAllWindows()
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    

def create_img(file_name, ti, tf):
    # Configuración del video
    cap = cv2.VideoCapture(file_name)
    fps = cap.get(cv2.CAP_PROP_FPS)                          # Frames por segundo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # Número total de frames

    # Calcular el frame inicial y final
    start_frame = int(ti * fps)
    end_frame = int(tf * fps)

    # Asegurarse de que los frames estén dentro del rango del video
    start_frame = min(max(0, start_frame), total_frames - 1)
    end_frame = min(max(0, end_frame), total_frames - 1)

    # Leer frames desde start_frame hasta end_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Lista donde guardaremos los frames en escala de grises del canal azul
    frames = []
     
    for frame_num in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        # Extraer el canal azul
        blue_channel = frame[:, :, 0]             # El 1er canal (índice 0) es el azul en OpenCV
        frames.append(blue_channel)

    # Convertir la lista de frames en un array de NumPy
    frames = np.array(frames)

    # Liberar el video
    cap.release()
    
    acumulado = np.zeros(frames[0].shape)
    for i in frames:
        acumulado = acumulado + i    
        
    mean_frame = acumulado/len(frames)
    
    return mean_frame#, frames, fps

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def autocorr(file_name, ti, step, tf, grafico):
    path = os.path.expanduser("~/Downloads/"+file_name)
    
    img_0 = create_img(path, ti, ti+step)
    img_0 = img_0 - np.mean(img_0)   
    
    mov_x = [0]
    mov_y = [0]
    t = 0
    i = 1

    while t <= tf: 
        img_1 = create_img(path, ti+(i * step) + 1, ti + ((i+1) * step))
        img_1 = img_1 - np.mean(img_1)
        
        # Correlación cruzada con FFT
        # corr = fftconvolve(img_0, img_1[::-1, ::-1], mode='same')
        corr = fftconvolve(img_1, img_0[::-1, ::-1], mode='same')
    
        # Pico de la correlación
        y_max, x_max = np.unravel_index(np.argmax(corr), corr.shape)
        
        # Centro
        y0, x0 = np.array(corr.shape) // 2
    
        # Desplazamiento
        dx = x_max - x0
        dy = y_max - y0
        
        mov_x.append(dx)
        mov_y.append(dy)
        
        t = ti + ((i+1) * step) # Actualizo el tiempo para que lo pueda comparar con el tiempo final
        i += 1                  # Actualizo el i para seguir iterando
    
    # Guardar mapa de autocorrelación en CSV
    pd.DataFrame({
        'x': np.tile(np.arange(corr.shape[1]), corr.shape[0]),
        'y': np.repeat(np.arange(corr.shape[0]), corr.shape[1]),
        'intensidad': corr.ravel()
    }).to_csv(f'/Users/tomas/Desktop/{grafico}_autocorrelacion.csv', index=False)
        
    mov_x = [abs(i) for i in mov_x]    
    mov_y = [abs(i) for i in mov_y]    
    
    duty_cycle_ida = [round(i*0.05, 2)*100 for i in range(len(mov_x)//2)]          # Ida
    duty_cycle_vuelta = [round(i*0.05, 2)*100 for i in range(len(mov_x)//2)][::-1] # Vuelta

    plt.close('all')
    fig, axs = plt.subplots(2, 1, figsize=(6, 9))
    fig.subplots_adjust(left=0.18,
                       right=0.96,
                       bottom=0.09,
                       top=0.99,
                       hspace=0.27,
                       wspace=0.26)
    
    axs[0].scatter(duty_cycle_ida, mov_x[:len(mov_x)//2], c="#A0522D", label='Ida')
    axs[0].scatter(duty_cycle_vuelta, mov_x[len(mov_x)//2:], c="#FF4500", label='Vuelta')
    
    axs[0].grid(linestyle=(0,(5, 3)), linewidth=1, alpha=.25)
    axs[0].set_ylabel(r'Desplazamiento en $\hat{x}$', fontsize=25, fontproperties=font_prop)
    axs[0].set_xlabel('Duty Cycle [%]', fontsize=25, fontproperties=font_prop)
    axs[0].set_xticks([0, 25, 50, 75, 100])
    axs[0].tick_params(axis='both', labelsize=18)
    axs[0].legend(fontsize=15, loc='lower right')


    axs[1].scatter(duty_cycle_ida, mov_y[:len(mov_y)//2], c="#A0522D")
    axs[1].scatter(duty_cycle_vuelta, mov_y[len(mov_y)//2:], c="#FF4500")
    
    axs[1].grid(linestyle=(0,(5, 3)), linewidth=1, alpha=.25)
    axs[1].set_ylabel(r'Desplazamiento en $\hat{y}$', fontsize=25, fontproperties=font_prop)
    axs[1].set_xlabel('Duty Cycle [%]', fontsize=25, fontproperties=font_prop)
    axs[1].set_xticks([0, 25, 50, 75, 100])
    axs[1].tick_params(axis='both', labelsize=15)
    
    plt.show()
    # plt.savefig(f'/Users/tomas/Desktop/{grafico}.png', bbox_inches='tight')
    
    
    fig, axs = plt.subplots(3, 1, figsize=(5, 10))
    fig.subplots_adjust(left=0.06,
                        right=0.97,
                        bottom=0.04,
                        top=0.99,
                        hspace=0.25,
                        wspace=0.26)    
    
    axs[0].imshow(img_0)
    axs[0].axvline(370, linestyle='--', c='r', alpha=.5)

    axs[1].imshow(img_1)
    axs[1].axvline(370, linestyle='--', c='k', alpha=.5)
    axs[1].axvline(370+dx, linestyle='--', c='r', alpha=.5)

    axs[2].imshow(corr)
    axs[2].axvline(x0, linestyle='--', c='r', alpha=.5)
    axs[2].axvline(x_max, linestyle='--', c='r', alpha=.5)
    # plt.savefig(f'/Users/tomas/Desktop/{grafico}_imagenes.png', bbox_inches='tight')

#%%

autocorr('WIN_20250407_13_29_49_Pro.mp4', 0, 4, 20, 'eje_Y')
