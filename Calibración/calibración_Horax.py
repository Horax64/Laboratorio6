import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd 

# 1. Configuración de rutas y parámetros
trayectorias_path = r'Analisis de video\Datos_tray\Barrido_1805_separado.csv'
data = pd.read_csv(trayectorias_path)
tiempos = data['t_0_video'].unique()




# plt.scatter(x_1,y_1)
# plt.xlabel('x [px]')
# plt.ylabel('y [px]')
# plt.title('Barrido cross-talk discreto (primera fila)')
# plt.show()

def lineal(x,a,b):
    return a*x + b

pendientes = []
ordenadas = []



# Sección cross-talk entre ejes

#for tiempo in tiempos:
#    mask = data['t_0_video'] == tiempo
#    x_traj = data[mask]['X'].values  
#    y_traj = data[mask]['Y'].values
#
#    popt, pcov = curve_fit(lineal,x_traj,y_traj,p0=[1,0])
#    pendientes.append(popt[0])
#    ordenadas.append(popt[1])
#
#pendiente_promedio = np.mean(pendientes)
#ordenada_promedio = np.mean(ordenadas)

# Código para promediar los clusters obtenidos en el barrido discreto 


def promediar_clusters_corregido(x_data, y_data, resolucion_x, min_frames):
    """
    Agrupa datos de un barrido utilizando un histograma de densidad espacial en 1D.
    Ideal para separar estados estacionarios de transitorios usando solo NumPy.
    """
    x = np.array(x_data)
    y = np.array(y_data)
    
    # 1. Definimos los bordes (bins) del histograma según la resolución pedida
    min_x, max_x = np.min(x), np.max(x)
    # Calculamos cuántos "cajoncitos" necesitamos para cubrir todo el barrido
    cantidad_bins = int(np.ceil((max_x - min_x) / resolucion_x))
    
    if cantidad_bins <= 0:
        return [], [], [], []
    
    # 2. np.histogram cuenta cuántos puntos caen en cada cajón espacial
    # counts: cantidad de puntos. edges: las coordenadas de los bordes del cajón
    counts, edges = np.histogram(x, bins=cantidad_bins)
    
    # 3. Filtramos: nos quedamos solo con los índices de los bins que tienen 
    # una cantidad de frames mayor a nuestro umbral (los picos estacionarios)
    bins_validos = np.where(counts >= min_frames)[0]
    
    if len(bins_validos) == 0:
        print("No se encontró ningún clúster con esos parámetros.")
        return [], [], [], []

    # 4. Agrupamos bins contiguos. Si un clúster físico es un poco ancho, 
    # puede ocupar 2 o 3 bins adyacentes. Si el índice salta, pasamos a otro clúster.
    saltos = np.where(np.diff(bins_validos) > 1)[0] + 1
    grupos_bins = np.split(bins_validos, saltos)
    
    x_promedios, y_promedios = [], []
    x_err, y_err = [], []
    
    # 5. Para cada grupo de bins (cada clúster detectado), extraemos los datos originales
    for grupo in grupos_bins:
        if len(grupo) == 0:
            continue
            
        # Tomamos el límite inferior del primer bin y el superior del último bin de este grupo
        limite_inf = edges[grupo[0]]
        limite_sup = edges[grupo[-1] + 1] # +1 para el borde derecho
        
        # Máscara booleana: rescatamos los puntos (X,Y) crudos que cayeron en este rango
        mascara = (x >= limite_inf) & (x <= limite_sup)
        puntos_x = x[mascara]
        puntos_y = y[mascara]
        
        # Procesamiento estadístico sobre el clúster limpio
        x_promedios.append(np.mean(puntos_x))
        y_promedios.append(np.mean(puntos_y))
        x_err.append(np.std(puntos_x))
        y_err.append(np.std(puntos_y))
        
    print(f"Método Histograma: Se aislaron {len(x_promedios)} clústers.")
    
    return np.array(x_promedios), np.array(y_promedios), np.array(x_err), np.array(y_err)

# --- Ejemplo de uso ---
# resolucion_x: Tamaño del "cajón" en tus unidades (ej: 0.1 micrómetros o píxeles)
# min_frames: Cantidad mínima de frames que el piezo tuvo que estar quieto en ese cajón

# x_mean, y_mean, x_std, y_std = promediar_clusters_histograma(x_track, y_track, resolucion_x=0.2, min_frames=10)

# --- Ejemplo de uso ---
# Suponiendo que ya cargaste tus datos de tracking del video (ej. con pd.read_csv o np.loadtxt)


umbral_x = 10  # Ajustá esto según el "paso" en micrómetros o píxeles de tu barrido


for tiempo in tiempos:
    print(tiempo)
    if tiempo > 0:
        mask = data['t_0_video'] == tiempo
        x_traj = data[mask]['X'].values  
        y_traj = data[mask]['Y'].values
        x_mean, y_mean, x_std, y_std = promediar_clusters_corregido(x_traj, y_traj,umbral_x,min_frames=5)

        plt.figure(figsize=(8, 6))
        plt.scatter(x_traj, y_traj, color='gray', alpha=0.3, label='Tracking crudo')
        plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt='ro', 
                        capsize=3, label='Centroides (promedio)')
        plt.xlabel('Desplazamiento X')
        plt.ylabel('Desplazamiento Y')
        plt.title('Promediado de Clústers de Tracking')
        plt.legend()
        plt.grid(True)
        plt.show()
    
dc_x = range(0,65535,3276)
dc_x = (1/65535)*np.array(dc_x)

len(x_mean)

print(len(x_mean))
print(len(dc_x))

# # Visualización rápida para ver que haya quedado joya






# Calibración de distancia
x_mean = x_mean*0.025239

coef1, cov1 = np.polyfit(dc_x[0:18],x_mean[0:18],1,cov='unscaled')
coef4, cov4 = np.polyfit(dc_x[0:18],x_mean[0:18],4,cov='unscaled')


print(cov4)
print(cov1)

def pol4(x,a,b,c,d,e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e


plt.plot(dc_x[0:18],lineal(dc_x[0:18],*coef1),'r',label='Ajuste polinomio grado 4')
plt.scatter(dc_x[0:18],x_mean[0:18])
plt.title('Histéresis de ida en x')
plt.xlabel('Duty cycle')
plt.ylabel('x[um]')
plt.grid(True)
plt.show()

