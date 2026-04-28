"""
@author: Tomás Obregón
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
from scipy.signal import fftconvolve
import cv2
import os
import pandas as pd
import time

# Modelo de ajuste 
def true_gaussian(dom, amp, ux, uy, a, b, c, offs):
    x,y = dom
    return amp * np.exp(a*((x-ux)**2) + b*(x-ux)*(y-uy) + c*((y-uy)**2)) + offs



class Video:
    """
    Clase Video
        path: ruta al archivo de video
        channel: canal a usar
        ti: tiempo inicial del recorte actual (en seg)
        tf: tiempo final del recorte actual (en seg)
        frames: lista de arrays de los frames del recorte actual np.arr[np.arr(h,l)] donde h es
                la altura del vídeo en px y l la longitud.
        fps: fps del recorte
        mean_frame: promedio de todos los frames del recorte [np.array(h,l)]

    """
    def __init__(self, path, channel):
        """
        Inicializa la clase fijando la ruta del video y el canal a usar (r,g, o b)
        """
        self.path = path
        self.channel = channel

        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps   


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
    def create_video(self, ti, tf):
        """
        Crea el recorte de vídeo a estar útlizando.
            ti: tiempo inicial en seg.
            tf: tiempo final en seg.
        Setea las variables 'tf', 'ti, 'frames' y 'fps'.
        """
        self.ti, self.tf = ti, tf
        
        # Configuración del video
        cap = cv2.VideoCapture(self.path)
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
            if self.channel == 'b':
                blue_channel = frame[:, :, 0]    # El 1er canal (índice 0) es el azul en OpenCV
                frames.append(blue_channel)
            elif self.channel == 'g':
                green_channel = frame[:, :, 1]   # El 2er canal (índice 1) es el verde en OpenCV
                frames.append(green_channel)
            elif self.channel == 'r':
                red_channel = frame[:, :, 2]     # El 3er canal (índice 2) es el rojo en OpenCV
                frames.append(red_channel)
            elif self.channel == 'all':    
                frames.append(frame)       # Todos los canales
    
        # Convertir la lista de frames en un array de NumPy
        self.frames = np.array(frames)
        self.fps = fps       
        
        # Liberar el video
        cap.release()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def get_frames(self,t0_recorte,tf_recorte):
        """
        Devuelve un fragmento del video, entre t0_recorte y tf_recorte
        """
        # Configuración para conseguir el frame deseado según su ubicación temporal

        frame_0 = int((t0_recorte-self.ti) * self.fps)
        frame_f = int((tf_recorte-self.ti) * self.fps)

        if frame_0 == frame_f:
            return([self.frames[frame_0]])
        else: return(self.frames[frame_0-1:frame_f])


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
    def average(self, frames):
        """
        Devuelve un frame (np.arr[h,l] con h: altura de las imagenesen px, l: longitud) 
        construido como un promedio punto a punto de todas las frames del video 
        actual menos el promedio de todos los puntos para todas las frames. 
        """
        acumulado = np.zeros(frames[0].shape)
        for i in frames:
            acumulado = acumulado + i    
            
        return (acumulado/len(frames)) - np.mean(acumulado/len(frames))
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def set_mean_frame(self, use_other = False, *frames):
        """
        Setea la variable mean_frame para ser el promedio del vídeo actual.
        Si se desea usar un frame distinto al mean_frame genera, use_other = True y *frames 
        debe ser un frame del video.
        """
        if use_other == True:
            self.mean_frame = self.average(frames[0])
        else: 
            self.mean_frame = self.average(self.frames)
        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def show_average_frame(self):
        """
        Muestra en un plot el estado actual del 'mean_frame'
        """

        if not hasattr(self, 'mean_frame'):
            print('mean_frame sin definir')

        else:     
            plt.figure(figsize=(10,5))
            plt.imshow(self.mean_frame)
            
            plt.xlabel(r'$\hat{x}$', fontsize=15)
            plt.ylabel(r'$\hat{y}$', fontsize=15)
            plt.tight_layout()
            plt.show()
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def show_frames(self):
        """
        Permite visualizar el recorte del vídeo actual. Es una visualización interactiva:
            'q' o 'Esc' para finalizar.
            'p' para pausar
            ',' para avanzar
            '.' para retroceder
        """
        
        index = 0
        paused = False
        running = True  # bandera para cortar el loop de forma controlada
    
        while running:
            if not paused:
                frame = self.frames[index]
                cv2.imshow('Video', frame)
                key = cv2.waitKey(int(1000 / self.fps)) & 0xFF
            else:
                key = cv2.waitKey(0) & 0xFF  # Espera indefinida
    
            # Lógica de control
            if key == ord('q') or key == 27:  # 'q' o ESC
                running = False
            elif key == ord('p'):
                paused = not paused
            elif key == ord(','):
                index = max(0, index - 10)
            elif key == ord('.'):
                index = min(len(self.frames) - 1, index + 10)
            elif not paused:
                index += 1
    
            if index >= len(self.frames):
                index = 0
    
        # Cierre seguro
        cv2.destroyAllWindows()
        cv2.waitKey(1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    def create_background(self, ti_bg, tf_bg, plot=False):
        """
        Crea el frame a utilizar como "background" a partir del cuál se mediran los
        desplazamientos. Si tf_bg y ti_bg son distintos, crea un frame promedio llamando
        a 'average()'.
            ti_bg: es el tiempo en segundos a partir del cual se tomará el promedio.
            tf_bg: es el tiempo en segundos hasta el cual se tomará el promedio.
        Si tf_bg es igual a tf_bg se utiliza el frame asociado a ese tiempo.
            plot: si es True, grafica el frame a usar como background.
        """
        self.ti_bg, self.tf_bg = ti_bg, tf_bg

        bg_frames = self.get_frames(ti_bg,tf_bg)

        self.background = self.average(bg_frames)

        if plot:
            plt.figure(figsize=(10,5))
            plt.imshow(self.background)
            
            plt.xlabel(r'$\hat{x}$', fontsize=15)
            plt.ylabel(r'$\hat{y}$', fontsize=15)
            plt.tight_layout()
            plt.show()
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def autocorr(self, mask, plot=False):
        """
        Calcula la autocorrelación entre 1 frame del vídeo (o promedio de frames de vídeo)
        y el background. El mapa de correlación debería devolver un máximo gaussiano que marca
        el punto que vamos a seguir.
            mask: tamaño en pixeles alrededor del máximo del mapa de correlación a seguir.
            plot: si es True devuelve una visualización el mapa de autocorrelación creado.
        Almacena en las variables x_mask, y_mask y z_mask las 3 variables del mapa de autocorrelación
        """
        self.mask=mask

        # Autocorrelacion
        # mean_frame se recorre en sentido inverso por la definición de convolución y correlación
        self.corr = fftconvolve(self.background, self.mean_frame[::-1, ::-1], mode='same')
        
        # Maximos
        self.y_max, self.x_max = np.unravel_index(np.argmax(self.corr), self.corr.shape)
        
        
        # Coordenadas X e Y
        x = np.arange(0, self.corr.shape[1])
        y = np.arange(0, self.corr.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Mascara
        self.x_mask = X[self.y_max-mask:self.y_max+mask+1, self.x_max-mask:self.x_max+mask+1] 
        self.y_mask = Y[self.y_max-mask:self.y_max+mask+1, self.x_max-mask:self.x_max+mask+1] 
        self.z_mask = self.corr[self.y_max-mask:self.y_max+mask+1, self.x_max-mask:self.x_max+mask+1] 

        if plot:
            plt.close('all')
            plt.figure(figsize=(10,5))
            plt.imshow(self.corr)
            
            plt.xlabel(r'$\hat{x}$', fontsize=15)
            plt.ylabel(r'$\hat{y}$', fontsize=15)
            plt.tight_layout()
            plt.show()
            time.sleep(0.1)
            plt.close('all')

        # if plot:
        #     plt.close('all')
        #     # Definir figura con gridspec
        #     fig = plt.figure(figsize=(10, 5))
        #     gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])  # gráfico izquierdo más ancho
            
        #     # Gráfico IZQUIERDO - ocupa dos filas
        #     ax1 = fig.add_subplot(gs[:, 0], projection='3d')
        #     ax1.plot_surface(self.x_mask, self.y_mask, self.z_mask, color='r', linewidth=0, antialiased=True, alpha=.8)
            
        #     ax1.set_xlabel(r'$\hat{x}$ [px]')
        #     ax1.set_ylabel(r'$\hat{y}$ [px]')
        #     ax1.set_zticks([])
        #     ax1.tick_params(axis='both', labelsize=10)
        #     ax1.set_box_aspect([self.corr.shape[1]/self.corr.shape[0], 1, 1])
        #     ax1.view_init(40, -75)
            
        #     # Gráfico SUPERIOR DERECHO
        #     ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        #     ax2.plot_surface(X, Y, self.corr, cmap='viridis', linewidth=0, antialiased=False, alpha=.15)
        #     ax2.plot_surface(self.x_mask, self.y_mask, self.z_mask, color='r', linewidth=0, antialiased=False)
            
        #     ax2.set_xlabel(r'$\hat{x}$ [px]')
        #     ax2.set_ylabel(r'$\hat{y}$ [px]')
        #     ax2.set_zticks([])
        #     ax2.tick_params(axis='both', labelsize=10)
        #     ax2.set_box_aspect([self.corr.shape[1]/self.corr.shape[0], 1, 1])
        #     ax2.view_init(40, -75)
            
        #     # Gráfico INFERIOR DERECHO
        #     ax3 = fig.add_subplot(gs[1, 1], projection='3d')
        #     ax3.plot_surface(X, Y, self.corr, cmap='viridis', linewidth=0, antialiased=False, alpha=.15)
        #     ax3.plot_surface(self.x_mask, self.y_mask, self.z_mask, color='r', linewidth=0, antialiased=False)
            
        #     ax3.set_xticks([])
        #     ax3.set_yticks([])
        #     ax3.set_zticks([])
        #     ax3.tick_params(axis='both', labelsize=10)
        #     ax3.set_box_aspect([self.corr.shape[1]/self.corr.shape[0], 1, 1])
        #     ax3.view_init(90, -90)
            
        #     plt.tight_layout()
        #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        #     plt.show()
                   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def fit(self, plot=False):
        """
        Ajusta el máximo de intensidad del mapa de correlación a una gausiana 2D para poder trackear
        el movimiento de la muestra.
            plot: si es True,
        ux, uy: posición del máximo de intensidad en x e y (en pixeles)
        err_ux, err_uy: errores del ajuste de la posición del máximo de intesidad
        """
        x_flat = self.x_mask.ravel()
        y_flat = self.y_mask.ravel()
        z_flat = self.z_mask.ravel()

        # Creo el meshgrid con una densidad a elección
        x_dense = np.linspace(min(x_flat), max(x_flat), 100)
        y_dense = np.linspace(min(y_flat), max(y_flat), 100)
        x_dense, y_dense = np.meshgrid(x_dense, y_dense)
        
        p0  = [np.max(self.corr)-np.min(self.corr), self.x_max, self.y_max, 0,0,0, np.max(self.corr)]         
        popt, pcov = op.curve_fit(true_gaussian, (x_flat, y_flat), z_flat, p0=p0, maxfev=10000)
        self.ux = popt[1]
        self.uy = popt[2] 
        self.err_ux = np.sqrt(pcov[1,1])
        self.err_uy = np.sqrt(pcov[2,2])

        z_dense = true_gaussian((x_dense, y_dense), *popt)
        
        # Residuos
        z_dense_para_residuos =  true_gaussian((self.x_mask, self.y_mask), *popt)
        res = self.z_mask - z_dense_para_residuos

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Grafico - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if plot:
            plt.close('all')
            fig = plt.figure(figsize=(8, 8))
            text = f'Segmento:\n{self.ti} s - {self.tf} s'
            fig.text(0.5, 0.5, text, ha='center', va='center', fontsize=16, weight='bold', color='black')


            # AX1 - - - - 
            ax1 = fig.add_subplot(2,2,1, projection='3d')

            # Rojo
            ax1.plot_surface(self.x_mask, self.y_mask, self.z_mask, color='r', alpha=.3)
            ax1.errorbar([self.x_max], [self.y_max], [max(z_flat)], xerr=2, yerr=2,
                          capsize=3, elinewidth=1, marker='x', color='red', markersize=5)

            # Verde
            ax1.errorbar([self.ux], [self.uy], [np.max(z_dense)], xerr=self.err_ux, yerr=self.err_uy,
                          capsize=3, elinewidth=1, marker='x', color='g', markersize=5)
            ax1.plot_surface(x_dense, y_dense, z_dense, color='g', alpha=.3)


            ax1.set_xlabel(r'$\hat{x}$ [px]')
            ax1.set_ylabel(r'$\hat{y}$ [px]')
            ax1.set_zticks([])
            ax1.set_box_aspect([self.z_mask.shape[1]/self.z_mask.shape[0], 1, 1])
            ax1.view_init(90, -90)


            # AX2 - - - - 
            ax2 = fig.add_subplot(2,2,2, projection='3d')

            # Rojo
            ax2.plot_surface(self.x_mask, self.y_mask, self.z_mask, color='r', alpha=.3)
            ax2.errorbar([self.x_max], [self.y_max], [max(z_flat)], xerr=2,
                          capsize=3, elinewidth=1, marker='x', color='red', markersize=5)

            # Verde
            ax2.errorbar([self.ux], [self.uy], [np.max(z_dense)], xerr=self.err_ux, 
                          capsize=3, elinewidth=1, marker='x', color='g', markersize=5)
            ax2.plot_surface(x_dense, y_dense, z_dense, color='g', alpha=.3)

            ax2.set_xlabel(r'$\hat{x}$ [px]')
            ax2.set_zlabel('Intensidad')
            ax2.set_yticks([])
            ax2.set_box_aspect([self.z_mask.shape[1]/self.z_mask.shape[0], 1, 1])
            ax2.view_init(0, -90)


            # AX3 - - - - 
            ax3 = fig.add_subplot(2,2,3, projection='3d')

            # Rojo
            ax3.plot_surface(self.x_mask, self.y_mask, self.z_mask, color='r', alpha=.3)
            ax3.errorbar([self.x_max], [self.y_max], [max(z_flat)], yerr=2,
                          capsize=3, elinewidth=1, marker='x', color='red', markersize=5)

            # Verde
            ax3.errorbar([self.ux], [self.uy], [np.max(z_dense)], yerr=self.err_uy,
                          capsize=3, elinewidth=1, marker='x', color='g', markersize=5)
            ax3.plot_surface(x_dense, y_dense, z_dense, color='g', alpha=.3)

            ax3.set_zlabel('Intensidad')
            ax3.set_ylabel(r'$\hat{y}$ [px]')
            ax3.set_xticks([])
            ax3.set_box_aspect([self.z_mask.shape[1]/self.z_mask.shape[0], 1, 1])
            ax3.view_init(0,0)

            # AX4 - - - - 
            ax4 = fig.add_subplot(2,2,4, projection='3d')
            ax4.plot_surface(self.x_mask, self.y_mask, res/1e5, color='g', alpha=.3)

            ax4.set_zlabel('Residuos (1e5)')
            ax4.set_ylabel(r'$\hat{y}$ [px]')
            ax4.set_xlabel(r'$\hat{x}$ [px]')
            ax4.set_box_aspect([self.z_mask.shape[1]/self.z_mask.shape[0], 1, 1])
            ax4.view_init(10, -10)

            plt.tight_layout()
            plt.show()
            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
    def hysteresis(self, mask, t0, tf, delta_t):        
        mov_x = [0]
        mov_y = [0] 
        err_mov_x = [0]
        err_mov_y = [0] 
        
        xmax, ymax = [0], [0]
        
        n_pasos = int((tf-t0)//delta_t)
        print(n_pasos)

        #autocorr_maps = []
        for i in range(n_pasos):
            if i == 0:
                self.create_background(t0,t0)
                frames_to_avg = self.get_frames(t0+delta_t,t0+delta_t)  
                self.set_mean_frame(True, frames_to_avg)
                self.autocorr(mask)
                self.fit(False)
                x0 = self.ux
                y0 = self.uy

                t = t0 + delta_t
            else:
                #self.create_background(t,t)
                frames_to_avg = self.get_frames(t+delta_t,t+delta_t)  
                self.set_mean_frame(True, frames_to_avg)
                self.autocorr(mask,True)
                self.fit(False)
                dx = abs(self.ux - x0)
                dy = abs(self.uy - y0)
                
                deltax = abs(self.x_max - x0)
                deltay = abs(self.y_max - y0)            
                xmax.append(deltax)
                ymax.append(deltay)
            
                
                #autocorr_maps.append(self.corr)
                mov_x.append(dx)
                mov_y.append(dy)
                err_mov_x.append(self.err_ux)
                err_mov_y.append(self.err_uy)
                
                t = t + delta_t
            
        print(mov_x)
        plt.scatter(range(n_pasos),mov_x)
        plt.show()

        # duty_cycle_ida = np.arange(0, 105, 5)          # Ida
        # mov_x_ida = mov_x[ : len(mov_x)//2+1]
        # mov_y_ida = mov_y[ : len(mov_x)//2+1]
        # err_mov_x_ida = err_mov_x[ : len(mov_x)//2+1]
        # err_mov_y_ida = err_mov_y[ : len(mov_x)//2+1]

        # duty_cycle_vuelta = np.arange(95, -5, -5)      # Vuelta
        # mov_x_vuelta = mov_x[len(mov_x)//2+1 : ]
        # mov_y_vuelta = mov_y[len(mov_x)//2+1 : ]
        # err_mov_x_vuelta = err_mov_x[len(mov_x)//2+1 : ]
        # err_mov_y_vuelta = err_mov_y[len(mov_x)//2+1 : ]
        
        
        # xmax_ida = xmax[ : len(mov_x)//2+1]
        # xmax_vuelta = xmax[len(mov_x)//2+1 : ]
                
        # ymax_ida = ymax[ : len(mov_x)//2+1]
        # ymax_vuelta = ymax[len(mov_x)//2+1 : ]
        
        # - - - - - - - - - - - - - - - - - - - - - 
        
        # plt.close('all')
        # fig = plt.figure(figsize=(8, 8))

        # # AX1 - - - - 
        # ax1 = fig.add_subplot(2,1,1)

        # ax1.errorbar(duty_cycle_ida, mov_x_ida, yerr=err_mov_x_ida,
        #              capsize = 3, elinewidth=2, linewidth=0, marker='.', markersize=5, 
        #              c="#A0522D", label='Ida')
        # ax1.errorbar(duty_cycle_vuelta, mov_x_vuelta, yerr=err_mov_x_vuelta,
        #              capsize = 3, elinewidth=2, linewidth=0, marker='.', markersize=5, 
        #              c="#FF4500", label='Vuelta')   

        # ax1.grid(linestyle=(0,(5, 3)), linewidth=1, alpha=.25)
        # ax1.set_ylabel(r'Desplazamiento en $\hat{x}$', fontsize=15)
        # ax1.set_xlabel('Duty Cycle [%]', fontsize=15)
        # ax1.set_xticks([0, 25, 50, 75, 100])
        # # ax1.set_yticks(np.arange(0,501,100))
        # ax1.tick_params(axis='both', labelsize=18)
        # ax1.legend(fontsize=15, loc='lower right')

        # # AX2 - - - - 
        # ax2 = fig.add_subplot(2,1,2)

        # ax2.errorbar(duty_cycle_ida, mov_y_ida, yerr=err_mov_y_ida,
        #              capsize = 3, elinewidth=2, linewidth=0, marker='.', markersize=5, 
        #              c="#A0522D", label='Ida')
        # ax2.errorbar(duty_cycle_vuelta, mov_y_vuelta, yerr=err_mov_y_vuelta,
        #              capsize = 3, elinewidth=2, linewidth=0, marker='.', markersize=5, 
        #              c="#FF4500", label='Vuelta')   

        # ax2.grid(linestyle=(0,(5, 3)), linewidth=1, alpha=.25)
        # ax2.set_ylabel(r'Desplazamiento en $\hat{y}$', fontsize=15)
        # ax2.set_xlabel('Duty Cycle [%]', fontsize=15)
        # ax2.set_xticks([0, 25, 50, 75, 100])
        # # ax2.set_yticks(np.arange(0,7,2))
        # ax2.tick_params(axis='both', labelsize=18)
        # ax2.legend(fontsize=15, loc='lower right')
        
        # fig.subplots_adjust(left=.125,
		# 				 bottom=.1,
		# 				 right=.95, 
		# 				 top=.95,
		# 				 wspace=0,
		# 				 hspace=.2)
        
        
        # fig = plt.figure(figsize=(8, 8))
        # # AX1 - - - - 
        # ax1 = fig.add_subplot(2,1,1)

        # ax1.errorbar(duty_cycle_ida, xmax_ida, yerr=np.full(len(duty_cycle_ida), 2),
        #              capsize = 3, elinewidth=2, linewidth=0, marker='.', markersize=5, 
        #              c="#AD5B35", label='Ida')
        # ax1.errorbar(duty_cycle_vuelta, xmax_vuelta, yerr=np.full(len(duty_cycle_vuelta), 2),
        #              capsize = 3, elinewidth=2, linewidth=0, marker='.', markersize=5, 
        #              c="#FF4500", label='Vuelta')   

        # ax1.grid(linestyle=(0,(5, 3)), linewidth=1, alpha=.25)
        # ax1.set_ylabel(r'Desplazamiento en $\hat{x}$', fontsize=15)
        # ax1.set_xlabel('Duty Cycle [%]', fontsize=15)
        # ax1.set_xticks([0, 25, 50, 75, 100])
        # # ax1.set_yticks(np.arange(0,501,100))
        # ax1.tick_params(axis='both', labelsize=18)
        # ax1.legend(fontsize=15, loc='lower right')

        # # AX2 - - - - 
        # ax2 = fig.add_subplot(2,1,2)

        # ax2.errorbar(duty_cycle_ida, ymax_ida, yerr=np.full(len(duty_cycle_ida), 2),
        #              capsize = 3, elinewidth=2, linewidth=0, marker='.', markersize=5, 
        #              c="#A0522D", label='Ida')
        # ax2.errorbar(duty_cycle_vuelta, ymax_vuelta, yerr=np.full(len(duty_cycle_vuelta), 2),
        #              capsize = 3, elinewidth=2, linewidth=0, marker='.', markersize=5, 
        #              c="#FF4500", label='Vuelta')   

        # ax2.grid(linestyle=(0,(5, 3)), linewidth=1, alpha=.25)
        # ax2.set_ylabel(r'Desplazamiento en $\hat{y}$', fontsize=15)
        # ax2.set_xlabel('Duty Cycle [%]', fontsize=15)
        # ax2.set_xticks([0, 25, 50, 75, 100])
        # # ax2.set_yticks(np.arange(0,7,2))
        # ax2.tick_params(axis='both', labelsize=18)
        # ax2.legend(fontsize=15, loc='lower right')
        
        # fig.subplots_adjust(left=.125,
		# 				 bottom=.1,
		# 				 right=.95, 
		# 				 top=.95,
		# 				 wspace=0,
		# 				 hspace=.2)
        
        # self.hyst = [mov_x, mov_y, err_mov_x, err_mov_y] #, autocorr_maps]


        
#%% 
file_name = r'C:\Users\LEC\Desktop\Garcia Crespo-Arias Ceci\Análisis de vídeo\Barrido_ida_x.mp4'
path = file_name

video = Video(path, 'b')

# # Creo el background con el frame inicial
# video.create_video(3, 3)
# video.create_background(3,3,plot=True)

# # Creo el video y el meanframe instantáneo a seguir para hacer una prueba de autocorr
# video.create_video(3+1/video.fps, 3+1/video.fps)
# video.set_mean_frame()
# video.show_average_frame()
# video.autocorr(30, True)

# Creo el video para hacer el análisis de histeresis
video.create_video(3, 90)
video.hysteresis(20,3,90,1/video.fps)
    
#%%
# df = pd.DataFrame({
#         'mov_x': video.hyst[0],
#         'mov_y': video.hyst[1],
#         'err_mov_x': video.hyst[2],
#         'err_mov_y': video.hyst[3]
#        })
# df.to_csv('histeresis.csv', index=False)

#%%
# """
# Testeo temporal de la curva de histeresis
# """

# file_name = r'C:\Users\LEC\Desktop\Garcia Crespo-Arias Ceci\Análisis de vídeo\Barrido_ida_x.mp4'
# path = file_name

# video = Video(path, 'b')
# video.create_video(3+1/video.fps, 3+1/video.fps)
# video.set_mean_frame()

# times = []
# for pasos in range(10,110,10):
#     t0 = time.time()
#     video.hysteresis(30,3,1/video.fps,pasos)
#     tf = time.time()
#     dt = tf-t0  
#     times.append(dt)
#     print(pasos)

# plt.scatter(range(10,110,10), times)
# plt.show()
# print(times)