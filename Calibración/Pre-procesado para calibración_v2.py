#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd 

#%%
trayectorias_path = r'Analisis de video\Datos_tray\Discreto_x_0506.csv'
data = pd.read_csv(trayectorias_path)
dc_y = range(0,65535,3276)
dc_x = range(0,65535,3276)

#Me doy cuenta que por la estructura de los datos es más fácil seprar por filas usando el salto en x que por tiempos
i=0
j=0
idx_saltos_de_fila = [0]
num_fila = []

while i < len(data)-1:
    num_fila.append(j)
    if np.abs(data['X'][i+1]-data['X'][i]) > 50: #Definimos la diferencia máxima entre el elemento actual y el próximo a ser considerada como salto
        idx_saltos_de_fila.append(i)
        j += 1
    i+=1
num_fila.append(j)

#Actualizamos los datos del dataframne
data['fila'] = num_fila

#Chequeamos el resultado de la separación
filas = [fila for fila in data['fila'].unique()]
for fila in filas:
    mask = data['fila'] == fila
    x_traj = data[mask]['X'].values
    y_traj = data[mask]['Y'].values
    plt.scatter(x_traj,y_traj)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.title(f'Chequeo general, separacion de fila')
plt.show()

 #%%
#Filtramos los pequeños outliers que parecen a todas luces ser errores del tracker
datos_a_eliminar = []
for fila in filas:
    data_de_fila = data[data['fila'] == fila]
    saltos_y = np.abs(np.diff(data_de_fila['X'].values, n=1))
    for i,salto in enumerate(saltos_y):
        if salto > 1: #Defino la diferencia mínima en y para ser considerado outlier
            datos_a_eliminar.append(data_de_fila.index[i+1])

data = data.drop(index = datos_a_eliminar)

for fila in filas:
    mask = data['fila'] == fila
    x_traj = data[mask]['X'].values
    y_traj = data[mask]['Y'].values
    plt.scatter(x_traj,y_traj)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.title(f'Chequeo general, filtrado outliers')
plt.show()
#%%
#Filtro para el caso del barrido en y, las posiciones donde el tracker (y el Nano) se rompió
mask = data['fila'] == filas[5]
y_traj = data[mask]['Y'].values
plt.scatter(range(0,285,1),y_traj[0:285])
plt.show()

data = data.drop(index = data[mask][285::].index)
for fila in filas:
    mask = data['fila'] == fila
    x_traj = data[mask]['X'].values
    y_traj = data[mask]['Y'].values
    plt.scatter(x_traj,y_traj)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.title(f'Chequeo general, filtrado outliers')
plt.show()
#%%
#Por ahora parece estar funcionando de forma más que adecuada, vamos a guardar los datos
file_name = r'Analisis de video\Datos_tray\Discreto_x_2905_correcciones_proc'
data.to_csv(f'{file_name}.csv', index=False)  
# %%
