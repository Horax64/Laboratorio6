"""
Método de calibración de parámetros pensado para usar el último script de calculo dcs
"""
#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd 

#%%
"""Configuración de rutas y visualización de una fila"""

trayectorias_path = r'Analisis de video\Datos_tray\Discreto_x_1805_proc.csv'
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
"""
Ajuste de cross-talk entre ejes
Tomamos un modelo lineal y encontramos Y = m*X
"""

def lineal(x,a,b):
    return a*x + b
pendientes = []
ordenadas = []

for fila in filas:
    mask = data['fila'] == fila
    x_traj = data[mask]['X'].values  
    y_traj = data[mask]['Y'].values

    popt, pcov = curve_fit(lineal,x_traj,y_traj,p0=[1,0])
    pendientes.append(popt[0])
    ordenadas.append(popt[1])

pendiente_promedio = np.mean(pendientes)
ordenada_promedio = np.mean(ordenadas)

print(pendiente_promedio)
print(ordenada_promedio)

plt.hist(pendientes)
plt.show()

#%%
"""
Ajuste de las no linealidades del barrido de x.
Vamos a ajustar (X_real = f(dc_x)).
"""
# Código para promediar los clusters obtenidos en el barrido discreto 

def promediar_clusters_corregido(x_data, y_data, umbral_x):
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
    
    # Imprimimos diagnóstico rápido para que veas qué hizo
    print(f"Detectados {len(x_promedios)} clústers (pasos) válidos.")
    
    return x_promedios, y_promedios, x_err, y_err

ajustes_nolineal_y = []

#%%


for fila in filas:
    try:
        x_track = data[data['fila']==fila]['X']
        y_track = data[data['fila']==fila]['Y']

        print(x_track)

        umbral_x = 1  # Umbral en pixeles para separar clusters (mucho mayor al "ruido" del tracker)

        x_mean, y_mean, x_std, y_std = promediar_clusters_corregido(x_track, y_track, umbral_x)

        print(len(x_mean))

        # # Visualización rápida para ver que haya quedado joya
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

        dc_x = np.linspace(0,1,20)
        # Calibración de distancia

        coef = np.polyfit(dc_x,x_mean,3)
        def pol(x,c,d,e,f):
            return  c*x**3 + d*x**2 + e*x + f

        # plt.plot(dc_x,pol(dc_x,*coef),'r',label='Ajuste polinomio grado 4')
        # plt.scatter(dc_x[1:-3],-np.diff(x_mean)[0:-3])
        # plt.title('Histéresis de ida en x')
        # plt.xlabel('Duty cycle')
        # plt.ylabel('x[um]')
        # plt.gca().invert_yaxis()
        # plt.grid(True)
    except:
        1   
plt.show()

#%%
cantidad_dcs = 21 #Es necesario tener en claro cuales fueron los dcs para cada punto.
                  #Asumimos que mandamos un array equiespaciado

for i,fila in enumerate(filas):
    x_track = data[data['fila']==fila]['X']
    y_track = data[data['fila']==fila]['Y']

    umbral_x = 1  # Ajustá esto según el "paso" en micrómetros o píxeles de tu barrido

    x_mean, y_mean, x_std, y_std = promediar_clusters_corregido(x_track, y_track, umbral_x)

    if len(x_mean) == cantidad_dcs:
        # #Visualización rápida para ver que haya quedado joya
        # plt.figure(figsize=(8, 6))
        # plt.scatter(x_track, y_track, color='gray', alpha=0.3, label='Tracking crudo')
        # plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt='ro', 
        #             capsize=3, label='Centroides (promedio)')
        # plt.xlabel('Desplazamiento X')
        # plt.ylabel('Desplazamiento Y')
        # plt.title(f'Promediado de Clústers de Tracking {i}')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        dc_x = range(0,65535,3276)
        dc_x = (1/65535)*np.array(dc_x)

        print(dc_x)
        # Calibración de distancia
        x_mean = x_mean*0.025239

        coef = np.polyfit(dc_x,x_mean,3)
        ajustes_nolineal_y.append((i,coef))
        def pol4(x,c,d,e,f):
            return  c*x**3 + d*x**2 + e*x + f

        # plt.plot(dc_x,pol4(dc_x,*coef),'r',label='Ajuste polinomio grado 4')
        # plt.scatter(dc_x,y_mean)
        # plt.title('Histéresis de ida en x')
        # plt.xlabel('Duty cycle')
        # plt.ylabel('x[um]')
        # plt.gca().invert_yaxis()
        # plt.grid(True)

print(ajustes_nolineal_y)

plt.show()


coef_principales = [coefs[0] for fila,coefs in ajustes_nolineal_y]
print(coef_principales)

# Juntamos el número con los elementos del array en una sola lista por fila
filas = [[numero] + list(array) for numero, array in ajustes_nolineal_y]

# Creamos el DataFrame
df = pd.DataFrame(filas)

# (Opcional) Si querés ponerle nombres a las columnas para que quede prolijo:
df.columns = ['Fila', 'a', 'b', 'c','d']

# Lo exportamos a .csv sin el índice automático de pandas
df.to_csv('ajuste_cubico_y.csv', index=False)
# x_track = data[data['fila']==0]['X']
# y_track = data[data['fila']==0]['Y']

# umbral_x = 1  # Ajustá esto según el "paso" en micrómetros o píxeles de tu barrido

# x_mean, y_mean, x_std, y_std = promediar_clusters_corregido(x_track, y_track, umbral_x)

# print(coef)

# plt.scatter(range(0,len(x_mean)),x_mean)
# plt.title('Saltos en posición en función del dc')
# plt.xlabel('Duty cycle
# plt.ylabel('\Delta x[um]')  
# plt.gca().invert_yaxis()
# plt.grid(True)
# plt.show()


pend_promedio_cross_x = -0.03461811951503092
ordenada_promedio_cross_x = 316.7424179043115

pend_promedio_cross_y = -0.07531236878047691
ordenada_promedio_cross_y = 1485.4739436704124