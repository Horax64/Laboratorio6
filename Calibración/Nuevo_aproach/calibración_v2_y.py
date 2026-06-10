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
"""Configuración de rutas y visualización de una columna."""

trayectorias_path = r'Analisis de video\Datos_tray\Discreto_y_2705_v2_proc.csv'
data = pd.read_csv(trayectorias_path)
columnas = data['columna'].unique()
col = 3

x_1 = data[data['columna']==columnas[col]]['X']
y_1 = data[data['columna']==columnas[col]]['Y']

plt.scatter(y_1,x_1)
plt.xlabel('y [px]')
plt.ylabel('x [px]')
plt.title(f'Barrido cross-talk discreto (col {col})')
plt.show()

#%%
"""Ajuste de cross-talk entre ejes.
Tomamos un modelo lineal para ajustar X = m*Y con Y y X en px
"""

def lineal(y,a,b):
    return a*y + b

ajustes_lineal_y = []
pendientes = []
ordenadas = []

for i,col in enumerate(columnas):
    mask = data['columna'] == col
    x_traj = data[mask]['X'].values  
    y_traj = data[mask]['Y'].values

    popt, pcov = curve_fit(lineal,y_traj,x_traj,p0=[1,0])
    ajustes_lineal_y.append([i,popt[0],popt[1]])

    # # Visualización del ajuste lineal
    # plt.title(f'Ajuste lineal para la columna {i}')
    # plt.scatter(y_traj,x_traj)
    # puntos_graf = np.linspace(min(y_traj),max(y_traj),1000)
    # plt.plot(puntos_graf,lineal(puntos_graf,*popt))
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
    dy_frame = np.abs(np.diff(y))
    
    # 2. El salto físico real de un duty cycle a otro será mucho mayor que 
    # el ruido de trackeo entre dos frames del mismo clúster estacionario.
    indices_corte = np.where(dy_frame > umbral_x)[0] + 1
    
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
    print(f"Detectados {len(y_promedios)} clústers (pasos) válidos.")
    
    return x_promedios, y_promedios, x_err, y_err

#%%
""" Visualización del resultado de los clusters para cada fila.
Esta sección ayuda a corregir errores de ajustes de la proxima sección.
"""
for col in columnas:
    x_track = data[data['columna']==col]['X']
    y_track = data[data['columna']==col]['Y']

    umbral_x = 1  # Umbral en pixeles para separar clusters (mucho mayor al "ruido" del tracker)

    x_mean, y_mean, x_std, y_std = promediar_clusters(x_track, y_track, umbral_x)

    # Visualización rápida para ver que hacemos en cada fila
    # plt.figure(figsize=(8, 6))
    # plt.scatter(x_track, y_track, color='gray', alpha=0.3, label='Tracking crudo')
    # plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt='ro', 
    #             capsize=3, label='Centroides (promedio)')
    # plt.xlabel('Desplazamiento X')
    # plt.ylabel('Desplazamiento Y')
    # plt.title(f'Promediado de Clústers de Tracking {col}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
#%%

"""Ajustes de las no linealidades en x para cada fila.
Buscamos V_x = f(X_real) y lo ajustamos con un polinomio de grado 3.
"""

cantidad_dcs = 21 #Es necesario tener en claro cuales fueron los dcs para cada punto.
                  #Asumimos que mandamos un array equiespaciado
ajustes_nolineal_y = []

for i,col in enumerate(columnas):
    x_track = data[data['columna']==col]['X']
    y_track = data[data['columna']==col]['Y']

    umbral_x = 1  # Umbral en pixeles para separar clusters (mucho mayor al "ruido" del tracker)

    x_mean, y_mean, x_std, y_std = promediar_clusters(x_track, y_track, umbral_x)

    if len(y_mean) == cantidad_dcs:
        dc_y = range(0,65535,3276)
        dc_y = (1/65535)*np.array(dc_y)

        # Calibración de distancia
        y_mean = y_mean*umppx

        coef = np.polyfit(y_mean,dc_y,3)
        ajustes_nolineal_y.append([i,coef])

        #Visualización para cada columna        
        pol = np.poly1d(coef)
        plt.plot(y_mean,pol(y_mean),'r',label='Ajuste polinomio grado 3')
        plt.scatter(y_mean,dc_y)
        plt.title('No linealidades ida y (dc_y vs y)')
        plt.ylabel('Duty cycle (Norm)')
        plt.xlabel('y[px]')
        plt.grid(True) 
        plt.show()
#%%
"""Guardado de coieficientes.
Guardamos los datos que conseguimos en un csv para las filas en las
que el ajuste pudo realizarse.
"""

# Juntamos el número con los elementos del array en una sola lista por fila
columnas_nolin = [[numero] + list(array) for numero, array in ajustes_nolineal_y]

# Coeficientes del polinomio: dc_y = a*y^3 + b*y^2 + c*y +d; dónde dc_y norm. y, x en um
df_nolin = pd.DataFrame(columnas_nolin)
df_nolin.columns = ['Columnas', 'a', 'b', 'c','d']
df_nolin.to_csv('ajuste_cubico_y_calv2.csv', index=False)

# Coeficientes del ajuste lineal: x = m*y + b; dónde y, x en px
columnas_lin = [[numero,m,b]  for numero, m, b in ajustes_lineal_y]
df_lin = pd.DataFrame(columnas_lin)
df_lin.columns = ['Columnas', 'm', 'b']
df_lin.to_csv('ajuste_lin_y_calv2.csv', index=False)
