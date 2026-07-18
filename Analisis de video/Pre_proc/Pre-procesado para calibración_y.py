import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd 
from tkinter import filedialog
from pathlib import Path


#%%
ruta_archivo = Path(filedialog.askopenfilename(
    title="Abrir video a trackear",
    defaultextension=".csv",
    initialdir=Path.home(),        
    filetypes=[("CSV","*.csv*"),("All files", "*.*")]
))

if not ruta_archivo:
    print("No se seleccionó ningún archivo. Saliendo.")
    exit()

data = pd.read_csv(ruta_archivo)

#Me doy cuenta que por la estructura de los datos es más fácil seprar por columnas usando el salto en x que por tiempos
i=0
j=0
idx_saltos_de_columna = [0]
num_columna = []

while i < len(data)-1:
    num_columna.append(j)
    if np.abs(data['Y'][i+1]-data['Y'][i]) > 50: #Definimos la diferencia máxima entre el elemento actual y el próximo a ser considerada como salto
        idx_saltos_de_columna.append(i)
        j += 1
    i+=1
num_columna.append(j)

#Actualizamos los datos del dataframne
data['columna'] = num_columna

#Chequeamos el resultado de la separación
columnas = [columna for columna in data['columna'].unique()]
for columna in columnas:
    mask = data['columna'] == columna
    x_traj = data[mask]['X'].values
    y_traj = data[mask]['Y'].values
    plt.scatter(x_traj,y_traj)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.title(f'Chequeo general, separacion de columna')
plt.show()

 #%%
#Filtramos los pequeños outliers que parecen a todas luces ser errores del tracker
datos_a_eliminar = []
for columna in columnas:
    data_de_columna = data[data['columna'] == columna]
    saltos_y = np.abs(np.diff(data_de_columna['Y'].values, n=1))
    for i,salto in enumerate(saltos_y):
        if salto > 1: #Defino la diferencia mínima en y para ser considerado outlier
            datos_a_eliminar.append(data_de_columna.index[i+1])

data = data.drop(index = datos_a_eliminar)

for columna in columnas:
    mask = data['columna'] == columna
    x_traj = data[mask]['X'].values
    y_traj = data[mask]['Y'].values
    plt.scatter(x_traj,y_traj)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.title(f'Chequeo general, filtrado outliers')
plt.show()
#%%
## Filtro para el caso del barrido en y, las posiciones donde el tracker (y el Nano) se rompió
# mask = data['columna'] == columnas[5]
# y_traj = data[mask]['Y'].values
# plt.scatter(range(0,285,1),y_traj[0:285])
# plt.show()

# data = data.drop(index = data[mask][285::].index)
# for columna in columnas:
#     mask = data['columna'] == columna
#     x_traj = data[mask]['X'].values
#     y_traj = data[mask]['Y'].values
#     plt.scatter(x_traj,y_traj)
#     plt.xlabel('x [px]')
#     plt.ylabel('y [px]')
#     plt.title(f'Chequeo general, filtrado outliers')
# plt.show()
#%%
#Por ahora parece estar funcionando de forma más que adecuada, vamos a guardar los datos
ruta_guardado = str(ruta_archivo).replace('.csv','') + '_proc.csv'
data.to_csv(ruta_guardado, index=False)  
print(f'Archivo guardado en {ruta_guardado}')
# %%
