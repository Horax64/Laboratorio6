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
    def setTemplate(path, n0, channel=0):
        # Permite al usuario seleccionar la partícula inicial de forma interactiva
        cap = cv.VideoCapture(path)
        cap.set(cv.CAP_PROP_POS_FRAMES, n0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            if channel == 0: 
                frame = frame[:, :, 0]  # 0->azul
            
            elif channel == 1:
                frame = frame[:, :, 1]  #1->verde
            
            elif channel == 2:
                frame = frame[:, :, 2]  #2->rojo
            
            else:                      
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #Sino hacemos un promedio
           
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
    def inicio(path, centro, ancho_t, ancho_v, n0, canal=0): # Añadimos canal
        cap = cv.VideoCapture(path)
        cap.set(cv.CAP_PROP_POS_FRAMES, n0)                                    
        ret, frame = cap.read()
        cap.release()
        
        if not ret: raise Exception("No se encontró el frame inicial.")
            
        # Seleccionamos el canal azul desde el inicio
        if canal in [0, 1, 2]:
            frame_proc = frame[:, :, canal]
        else:
            frame_proc = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        y, x = centro[1], centro[0]
        ht, wt = ancho_t[1], ancho_t[0]
        hv, wv = ancho_v[1], ancho_v[0]

        # Template y Obs deben venir del MISMO canal que usará corr()
        template = frame_proc[y-ht:y+ht, x-wt:x+wt]
        obs = frame_proc[y-hv:y+hv, x-wv:x+wv]
        
        return template, obs
        
    @staticmethod
    def corr(path, template, obs, centro, tiempo, duracion, canal):
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
       
        # Parámetros de predicción
        umbral_confianza = 0.65
        memoria_velocidad = 5 # Cuántos frames usamos para calcular la inercia
        vx, vy = 0, 0         # Componentes del vector de velocidad actual
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > (duracion[1] - duracion[0]): break
            frame_idx += 1
            
            filt_frame = frame[:, :, canal]
            
            # 1. Aplicar predicción si ya tenemos datos previos
            # Si el motor está en medio de un salto, esto "mueve" la caja proactivamente
            if len(x_vec) > memoria_velocidad:
                vx = x_vec[-1] - x_vec[-memoria_velocidad]
                vy = y_vec[-1] - y_vec[-memoria_velocidad]
                # Normalizamos la velocidad por frame
                vx /= memoria_velocidad
                vy /= memoria_velocidad

            # 2. Definir área de observación basada en la posición actual (x, y)
            y_min, y_max = max(0, y - h_v), min(frame.shape[0], y + h_v)
            x_min, x_max = max(0, x - w_v), min(frame.shape[1], x + w_v)
            A = filt_frame[y_min:y_max, x_min:x_max]

            # 3. Correlación
            res = cv.matchTemplate(A, template, cv.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc_raw = cv.minMaxLoc(res)

            if max_val > umbral_confianza:
                # --- MODO TRACKING ACTIVO ---
                estado = "TRACKING"
                color_box = (0, 255, 0)
                
                x_sub, y_sub = interpolacion_subpixel(res, max_loc_raw[1], max_loc_raw[0])
                x_real = x_min + x_sub + w_t
                y_real = y_min + y_sub + h_t
                
                # Actualizamos x, y para el siguiente frame
                x, y = int(x_real), int(y_real)
            else:
                # --- MODO PREDICCIÓN (EL SALTO) ---
                estado = "PREDICCION POR INERCIA"
                color_box = (255, 0, 255) # Magenta para indicar predicción
                
                # En lugar de quedarnos quietos, aplicamos la velocidad calculada
                x_real = x_vec[-1] + vx
                y_real = y_vec[-1] + vy
                
                # Desplazamos la ventana de búsqueda para el próximo frame
                # Esto es lo que permite "encontrar" la partícula tras el salto
                x, y = int(x_real), int(y_real)
                print(f"Frame {frame_idx}: Salto detectado. Predicción: dx={vx:.2f}")

            x_vec.append(x_real)
            y_vec.append(y_real)

            # --- VISUALIZACIÓN ---
            img_disp = frame.copy()
            # Dibujar ventana de búsqueda (cian)
            cv.rectangle(img_disp, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
            # Dibujar partícula (Verde/Magenta)
            cv.rectangle(img_disp, (int(x_real-w_t), int(y_real-h_t)), 
                         (int(x_real+w_t), int(y_real+h_t)), color_box, 2)
            
            # Flecha de vector velocidad (para ver hacia dónde predice)
            if estado == "TRACKING":
                cv.arrowedLine(img_disp, (int(x_real), int(y_real)), 
                               (int(x_real + vx*10), int(y_real + vy*10)), (255, 255, 0), 2)

            cv.putText(img_disp, f"Estado: {estado}", (20, 30), 1, 1.2, color_box, 2)
            cv.imshow("Tracker Predictivo - LEC", img_disp)
            if cv.waitKey(tiempo) & 0xFF == ord("q"): break
        
        cap.release()
        cv.destroyAllWindows()
        for _ in range(4): cv.waitKey(1) # Previene crasheo al finalizar
                                    
        return x_vec, y_vec