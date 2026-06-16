"""
Método de calibración de parámetros pensado para usar el último script de calculo dcs.
"""
#%%
"""Importación de librerías."""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd 
import warnings

# Forzar a Python a lanzar un error al fallar el ajuste
warnings.filterwarnings('error', category=np.exceptions.RankWarning)

# Conversión de pixeles a micrones
umppx = 0.025239
#%%
"""Configuración de rutas y visualización de una fila."""
file = 'Discreto_cali_x_1606'
trayectorias_path = fr'Analisis de video\Datos_tray\{file}_proc.csv'
data = pd.read_csv(trayectorias_path)
filas = data['fila'].unique()
fila = 3

x_1 = data[data['fila']==filas[fila]]['X']
y_1 = data[data['fila']==filas[fila]]['Y']

plt.scatter(x_1,y_1)
plt.xlabel('x [px]')
plt.ylabel('y [px]')
plt.title(f'Barrido cross-talk discreto (fila {fila})')
plt.show()

#%%
"""Ajuste de cross-talk entre ejes.
Tomamos un modelo lineal para ajustar Y = m*X con Y y X en px
"""

def lineal(x,a,b):
    return a*x + b

ajustes_lineal_x = []
pendientes = []
ordenadas = []

for i,fila in enumerate(filas):
    mask = data['fila'] == fila
    x_traj = data[mask]['X'].values  
    y_traj = data[mask]['Y'].values

    popt, pcov = curve_fit(lineal,x_traj,y_traj,p0=[1,0])
    ajustes_lineal_x.append([i,popt[0],popt[1]])

    # # Visualización del ajuste lineal
    # plt.title(f'Ajuste lineal para la fila {i}')
    # plt.scatter(x_traj,y_traj)
    # puntos_graf = np.linspace(min(x_traj),max(x_traj),1000)
    # plt.plot(np.linspace(min(x_traj),max(x_traj),1000),lineal(puntos_graf,*popt))
    # plt.show()

## Si queremos podemos visualizar la distribución de las pendientes halladas para todas las filas

# ajustes_array = np.array(ajustes_lineal_x)
# pendiente = ajustes_array[:,1]
# ordenadas = ajustes_array[:,2]
# pendiente_promedio = np.mean(pendientes)
# ordenada_promedio = np.mean(ordenadas)

# print(f'La pendiente promedio es: {pendiente_promedio}')
# plt.title('Distribución de pendientes')
# plt.hist(pendientes)
# plt.xlabel('Pendiente (px/px)')
# plt.show()

#%%
"""Código para promediar los clusters obtenidos en el barrido discreto y obtener puntos relacionados con cada duty cycle
"""

def promediar_clusters(x_data, y_data, umbral_x):
    """
    Agrupa datos de tracking temporal basándose en saltos bruscos frame a frame.
    No ordena los datos, preservando la evolución temporal del barrido.
    """
    x = np.array(x_data)
    y = np.array(y_data)
    
    # 1. Calculamos la derivada discreta frame a frame.
    # Le ponemos np.abs() para que detecte saltos tanto en la ida como en la vuelta del barrido.
    dx_frame = np.abs(np.diff(x))
    
    # 2. El salto físico real de un duty cycle a otro será mucho mayor que 
    # el ruido de trackeo entre dos frames del mismo clúster estacionario.
    indices_corte = np.where(dx_frame > umbral_x)[0] + 1
    
    # 3. Partimos los arrays temporalmente
    clusters_x = np.split(x, indices_corte)
    clusters_y = np.split(y, indices_corte)
    
    # 4. Filtro de robustez: ignoramos "clústers" que tengan muy pocos puntos
    min_frames_por_paso = 5 
    
    x_promedios = np.array([np.mean(c) for c in clusters_x if len(c) > min_frames_por_paso])
    y_promedios = np.array([np.mean(c) for c in clusters_y if len(c) > min_frames_por_paso])
    
    x_err = np.array([np.std(c) for c in clusters_x if len(c) > min_frames_por_paso])
    y_err = np.array([np.std(c) for c in clusters_y if len(c) > min_frames_por_paso])
    
    # Imprimimos diagnóstico rápido para ver que hizo
    print(f"Detectados {len(x_promedios)} clústers (pasos) válidos.")
    
    return x_promedios, y_promedios, x_err, y_err

#%%
""" Visualización del resultado de los clusters para cada fila.
Esta sección ayuda a corregir errores de ajustes de la proxima sección.
"""
for fila in filas:
    x_track = data[data['fila']==fila]['X']
    y_track = data[data['fila']==fila]['Y']

    umbral_x = 1  # Umbral en pixeles para separar clusters (mucho mayor al "ruido" del tracker)

    x_mean, y_mean, x_std, y_std = promediar_clusters(x_track, y_track, umbral_x)

    # # Visualización rápida para ver que hacemos en cada fila
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_track, y_track, color='gray', alpha=0.3, label='Tracking crudo')
    # plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt='ro', 
    #             capsize=3, label='Centroides (promedio)')
    # plt.xlabel('Desplazamiento X')
    # plt.ylabel('Desplazamiento Y')
    # plt.title(f'Promediado de Clústers de Tracking {fila}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
#%%

"""Ajustes de las no linealidades en x para cada fila.
Buscamos V_x = f(X_real) y lo ajustamos con un polinomio de grado 3.
"""

cantidad_dcs = 16 #Es necesario tener en claro cuales fueron los dcs para cada punto.
                  #Asumimos que mandamos un array equiespaciado
ajustes_nolineal_x = []

for i,fila in enumerate(filas):
    x_track = data[data['fila']==fila]['X']
    y_track = data[data['fila']==fila]['Y']

    umbral_x = 1  # Umbral en pixeles para separar clusters (mucho mayor al "ruido" del tracker)

    x_mean, y_mean, x_std, y_std = promediar_clusters(x_track, y_track, umbral_x)

    if len(x_mean) == cantidad_dcs:
        dc_x = np.linspace(0,65535,cantidad_dcs)
        dc_x = (1/65535)*np.array(dc_x)

        # Calibración de distancia
        x_mean = x_mean*umppx

        coef = np.polyfit(dc_x,x_mean,3)
        ajustes_nolineal_x.append([i,coef])

        pol = np.poly1d(coef)

        #Visualización para cada fila        
        plt.plot(dc_x,pol(dc_x),'r',label='Ajuste polinomio grado 3')
        plt.scatter(dc_x,x_mean) #Corregir!!!
        plt.title('No linealidades ida x (dc_x vs x)')
        plt.xlabel('Duty cycle (Norm)')
        plt.ylabel('x[px]')
        plt.grid(True) 
        plt.show()
#%%
"""Guardado de coieficientes.
Guardamos los datos que conseguimos en un csv para las filas en las
que el ajuste pudo realizarse.
"""

# Juntamos el número con los elementos del array en una sola lista por fila
filas_nolin = [[numero] + list(array) for numero, array in ajustes_nolineal_x]

# Coeficientes del polinomio: dc_x = a*x^3 + b*x^2 + c*x +d; dónde dc_x norm. y, x en um
df_nolin = pd.DataFrame(filas_nolin)
df_nolin.columns = ['Fila', 'a', 'b', 'c','d']
df_nolin.to_csv('ajuste_cubico_x_calv1_1606.csv', index=False)

# Coeficientes del ajuste lineal: y = m*x + b; dónde x, y en px
filas_lin = [[numero,m,b]  for numero, m, b in ajustes_lineal_x]
df_lin = pd.DataFrame(filas_lin)
df_lin.columns = ['Fila', 'm', 'b']
df_lin.to_csv('ajuste_lin_x_calv1_1606.csv', index=False)

# %%
