import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd 

trayectorias_path = r'Prueba_separación_automática.csv'
data = pd.read_csv(trayectorias_path)
tiempos = [tiempo for tiempo in data['t_0_video'].unique() if tiempo >= 0]

# Recosntruimos y limpiamos las trayectorias de forma adecuada
tiempos_a_corregir = []

for tiempo in tiempos:
    mask = data['t_0_video'] == tiempo
    x_traj = data[mask]['X'].values
    y_traj = data[mask]['Y'].values
    plt.scatter(x_traj,y_traj)
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.title(f'Chequeo general')
plt.show()

for i,tiempo in enumerate(tiempos):
    mask = data['t_0_video'] == tiempo
    x_traj = data[mask]['X'].values
    y_traj = data[mask]['Y'].values

    if y_traj.max() - y_traj.min() < 13:    
        # plt.scatter(x_traj,y_traj)
        # plt.xlabel('x [px]')
        # plt.ylabel('y [px]')
        # plt.title(f'Chequeo de fila con problemas fila {i}')
        # plt.show()

        try: 
            mask = data['t_0_video'] == tiempos[i+1]
            x_next = data[mask]['X'].values  
            y_next = data[mask]['Y'].values
            # plt.scatter(x_next,y_next)
            # plt.xlabel('x [px]')
            # plt.ylabel('y [px]')
            # plt.title(f'Siguiente fila')
            # plt.show()

            if y_next[0] - y_traj[-1] < 3:
                x_traj_fix = np.concatenate((x_traj,x_next),axis=None)
                y_traj_fix = np.concatenate((y_traj, y_next),axis=None)

                # plt.scatter(x_traj_fix,y_traj_fix)
                # plt.xlabel('x [px]')
                # plt.ylabel('y [px]')
                # plt.title(f'Trayectoria corregida')
                # plt.show()

                tiempos_a_corregir.append(tiempos[i+1])
        except:
            print('Ultima fila')

print(tiempos_a_corregir)