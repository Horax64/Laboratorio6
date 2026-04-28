# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from skimage.feature import peak_local_max

def max_cercano(maximos, maximo_viejo):
    # Calcula la distancia euclidiana entre máximos locales y la posición anterior
    # Garantiza que el tracker siga la misma partícula y no salte a ruido cercano.
    distancias = [np.linalg.norm(maximos[i] - maximo_viejo) for i in range(len(maximos))]
    return maximos[np.argmin(distancias)]  

def interpolacion_subpixel(matriz_corr, y_int, x_int):
    """
    Calcula la posición sub-píxel exacta usando un ajuste parabólico 2D separable.
    """
    h, w = matriz_corr.shape
    
    # Suposición lógica: Si el máximo detectado está exactamente en el borde 
    # de la matriz de búsqueda, no tenemos vecinos para hacer la interpolación. 
    # En ese caso límite, devolvemos la coordenada entera cruda.
    if x_int == 0 or x_int == w - 1 or y_int == 0 or y_int == h - 1:
        return float(x_int), float(y_int)
        
    # --- Interpolar en el eje X ---
    R_x_menos = matriz_corr[y_int, x_int - 1]
    R_0 = matriz_corr[y_int, x_int]
    R_x_mas = matriz_corr[y_int, x_int + 1]
    
    denominador_x = 2 * (R_x_menos - 2 * R_0 + R_x_mas)
    # Evitamos error de división por cero si la zona es perfectamente plana
    if denominador_x != 0:
        delta_x = (R_x_menos - R_x_mas) / denominador_x
    else:
        delta_x = 0.0
        
    # --- Interpolar en el eje Y ---
    R_y_menos = matriz_corr[y_int - 1, x_int]
    R_y_mas = matriz_corr[y_int + 1, x_int]
    
    denominador_y = 2 * (R_y_menos - 2 * R_0 + R_y_mas)
    if denominador_y != 0:
        delta_y = (R_y_menos - R_y_mas) / denominador_y
    else:
        delta_y = 0.0
        
    return x_int + delta_x, y_int + delta_y

class tracker:
    
    @staticmethod
    def fps(path):
        # Abre el video solo para extraer el metadato de los fotogramas por segundo
        cap = cv.VideoCapture(path)
        fps_val = cap.get(cv.CAP_PROP_FPS)
        cap.release()
        return fps_val

    @staticmethod
    def setTemplate(path, n0):
        # Permite al usuario seleccionar la partícula inicial de forma interactiva
        cap = cv.VideoCapture(path)
        cap.set(cv.CAP_PROP_POS_FRAMES, n0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            bbox = cv.selectROI("Seleccion de particula", frame)
            
            # Forzamos vaciado de la cola de eventos visuales para evitar crasheos
            cv.destroyAllWindows()
            for _ in range(4):
                cv.waitKey(1)             
            
            (c, f, w, h) = [int(a) for a in bbox]  
            if w == 0 or h == 0:
                raise ValueError("No se seleccionó ninguna ROI válida.")
                
            centro = [int(c + w/2), int(f + h/2)] 
            ancho_t = [int(w/2), int(h/2)] 
            return centro, ancho_t
        else:
            raise Exception("No se encontró el video")

    @staticmethod
    def inicio(path, centro, ancho_t, ancho_v, n0):
        # Dibuja y recorta las matrices iniciales del Template y el Área de Búsqueda
        cap = cv.VideoCapture(path)
        cap.set(cv.CAP_PROP_POS_FRAMES, n0)                                    
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("No se encontró el frame inicial.")
            
        imag = frame.copy()
        cv.rectangle(imag, (centro[0]-ancho_t[0], centro[1]-ancho_t[1]), (centro[0]+ancho_t[0], centro[1]+ancho_t[1]), [0,0,0], 2) 
        template = cv.cvtColor(frame[centro[1]-ancho_t[1]:centro[1]+ancho_t[1], centro[0]-ancho_t[0]:centro[0]+ancho_t[0]], cv.COLOR_BGR2GRAY)
        
        cv.rectangle(imag, (centro[0]-ancho_v[0], centro[1]-ancho_v[1]), (centro[0]+ancho_v[0], centro[1]+ancho_v[1]), [0,0,255], 1)
        obs = cv.cvtColor(frame[centro[1]-ancho_v[1]:centro[1]+ancho_v[1], centro[0]-ancho_v[0]:centro[0]+ancho_v[0]], cv.COLOR_BGR2GRAY)
        
        cv.imshow("trackeo", imag)            
        cv.waitKey(1000) # Muestra la configuración inicial 1 segundo
        cv.destroyAllWindows()
        return template, obs
        
    @staticmethod
    def corr(path, template, obs, centro, tiempo, duracion):
        cap = cv.VideoCapture(path) 
        cap.set(cv.CAP_PROP_POS_FRAMES, duracion[0])   
        
        # Condiciones geométricas iniciales
        h_t, w_t = [int(i/2) for i in template.shape]        
        h_v, w_v = [int(i/2) for i in obs.shape]      
        d_h, d_w = h_v - h_t, w_v - w_t                    
        
        x, y = centro[0], centro[1]                     
        max_loc = [w_v, h_v] # Centro local relativo al área de observación
                
        fallos = 0                   
        frame_idx = 0     
        x_vec, y_vec = [], []
        
        A = obs.copy()        
        method = cv.TM_CCOEFF_NORMED # EFICIENCIA: Se define fuera del bucle
        
        while True:
            ret, frame = cap.read()    
            if not ret or frame_idx > (duracion[1] - duracion[0]):
                break
            
            frame_idx += 1                      
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  
            
            # CORRELACIÓN CRUZADA (Usa FFT bajo el capó)
            result = cv.matchTemplate(A, template, method) 
            
            try:
                # 1. Encontramos el máximo entero matricial [Fila, Columna] -> [Y, X]
                maximos = peak_local_max(result, min_distance=5, threshold_rel=0.3)
                max_loc_entero = max_cercano(maximos, max_loc) 
                
                # 2. Aplicamos la corrección matemática de sub-píxel
                x_sub, y_sub = interpolacion_subpixel(result, max_loc_entero[0], max_loc_entero[1])                   
                
                # 3. CORRECCIÓN 1: Mantenemos el formato matricial [Y, X] para no cruzar ejes
                max_loc = [int(round(y_sub)), int(round(x_sub))]
                
            except ValueError:
                fallos += 1
                if fallos > 5:
                    print(f"Se detuvo el trackeo por pérdida de foco en frame {frame_idx}")
                    break   
                continue 
                                    
            # 4. ACTUALIZACIÓN DE COORDENADAS GLOBALES
            # max_loc[1] es X_sub, max_loc[0] es Y_sub
            upper_left = (max_loc[1] + x - w_v, max_loc[0] + y - h_v) 
            
            # Calculamos y guardamos la posición decimal súper precisa
            x_real = (x_sub + x - w_v) + w_t
            y_real = (y_sub + y - h_v) + h_t
            
            x_vec.append(x_real)   
            y_vec.append(y_real)  
            
            # 5. CORRECCIÓN 2: Actualizamos el centro para el frame siguiente.
            # Sin esto, la caja nunca sigue a la partícula.
            x = int(upper_left[0]) + w_t
            y = int(upper_left[1]) + h_t
            
            try:
                upper_left_b = (upper_left[0] - d_w, upper_left[1] - d_h) 
                bottom_right_b = (upper_left_b[0] + 2*w_v, upper_left_b[1] + 2*h_v)
                A = gray[upper_left_b[1]:bottom_right_b[1], upper_left_b[0]:bottom_right_b[0]]                                
            except IndexError:
                print(f"Área de observación fuera de los límites de la imagen en frame {frame_idx}")
                break 
            
            x, y = int(upper_left[0]) + w_t, int(upper_left[1]) + h_t
            x_vec.append(x)   
            y_vec.append(y)   
            
            # VISUALIZACIÓN
            img_disp = frame.copy()
            bottom_right = (upper_left[0] + 2*w_t, upper_left[1] + 2*h_t) 
            cv.rectangle(img_disp, upper_left, bottom_right, [0,0,0], 2) 
            cv.rectangle(img_disp, upper_left_b, bottom_right_b, [0,0,255], 1)
            cv.imshow("trackeo", img_disp)

            if cv.waitKey(tiempo) & 0xFF == ord("q"): 
                break                    
        
        cap.release()
        cv.destroyAllWindows()
        for _ in range(4): cv.waitKey(1) # Previene crasheo al finalizar
                                    
        return x_vec, y_vec