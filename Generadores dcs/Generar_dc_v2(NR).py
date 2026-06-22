import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# ---------------------------------------------------------
# 1. CARGA Y DEFINICIÓN DEL MODELO DIRECTO
# ---------------------------------------------------------
file = 'calv1_1906'

try:
    datos_cross_x  = pd.read_csv(rf'Calibración\Aproach_NR\ajuste_lin_x_{file}.csv') 
    datos_cross_y  = pd.read_csv(rf'Calibración\Aproach_NR\ajuste_lin_y_{file}.csv') 

    promedio_cross_x = [datos_cross_x['m'].mean(),0]
    promedio_cross_y = [datos_cross_y['m'].mean(),0]

    datos_ajuste_x = pd.read_csv(rf'Calibración\Aproach_NR\ajuste_cubico_x_{file}.csv') 
    datos_ajuste_y = pd.read_csv(rf'Calibración\Aproach_NR\ajuste_cubico_y_{file}.csv')

    promedio_x = datos_ajuste_x.mean()[1::]
    coefs_x = [coef for coef in promedio_x] 
    coefs_x[-1] = 0

    promedio_y = datos_ajuste_y.mean()[1::]
    coefs_y = [coef for coef in promedio_y] 
    coefs_y[-1] = 0
except FileNotFoundError:
    print("No se encontraron los CSV. Usando coeficientes cúbicos de prueba...")
    coefs_x = [2.0, -0.5, 4.0, 0.0]  
    coefs_y = [1.8, -0.4, 3.8, 0.0]  
    promedio_cross_x = [0.05, 0]
    promedio_cross_y = [0.04, 0]

cross_x = np.poly1d(promedio_cross_x)
cross_y = np.poly1d(promedio_cross_y)

polinomio_x = np.poly1d(coefs_x)
polinomio_y = np.poly1d(coefs_y)

d_polinomio_x = np.polyder(polinomio_x)
d_polinomio_y = np.polyder(polinomio_y)

# ---------------------------------------------------------
# 2. MÉTODOS NUMÉRICOS (NEWTON-RAPHSON)
# ---------------------------------------------------------

def NewtonRaphsonIndividual(dc_x_guess, dc_y_guess, X, Y):
    Px = polinomio_x(dc_x_guess)
    Py = polinomio_y(dc_y_guess)
    
    # Modelo coherente (se usa + en ambos)
    F_x = Px + cross_x(Py) - X
    F_y = Py + cross_y(Px) - Y

    dPx = d_polinomio_x(dc_x_guess)
    dPy = d_polinomio_y(dc_y_guess)

    dFxdx = dPx
    dFxdy = promedio_cross_x[0] * dPy       
    dFydx = promedio_cross_y[0] * dPx       
    dFydy = dPy

    det_J = dFxdx * dFydy - dFxdy * dFydx
    
    if abs(det_J) < 1e-12:
        return dc_x_guess, dc_y_guess
    
    paso_x = (F_x * dFydy - F_y * dFxdy) / det_J
    paso_y = (-F_x * dFydx + F_y * dFxdx) / det_J

    max_step = 0.05 
    paso_x = max(-max_step, min(max_step, paso_x))
    paso_y = max(-max_step, min(max_step, paso_y))

    dc_x_obtenido = dc_x_guess - paso_x
    dc_y_obtenido = dc_y_guess - paso_y

    return dc_x_obtenido, dc_y_obtenido


def IteracionesNR(n, dc_x_guess, dc_y_guess, X, Y, tol=1e-6):
    x, y = dc_x_guess, dc_y_guess
    for _ in range(n):
         x_new, y_new = NewtonRaphsonIndividual(x, y, X, Y)
         if abs(x_new - x) < tol and abs(y_new - y) < tol:
             return x_new, y_new
         x, y = x_new, y_new
    return x, y

def desplazamientos(dcx_inicial, dcx_final, dcy_inicial, dcy_final, paso_um_x, paso_um_y):
    posicion_inicial_x = polinomio_x(dcx_inicial) + cross_x(polinomio_y(dcy_inicial))
    posicion_inicial_y = polinomio_y(dcy_inicial) + cross_y(polinomio_x(dcx_inicial))
    
    posicion_final_x   = polinomio_x(dcx_final) + cross_x(polinomio_y(dcy_final))
    posicion_final_y   = polinomio_y(dcy_final) + cross_y(polinomio_x(dcx_final))

    delta_x = posicion_final_x - posicion_inicial_x
    delta_y = posicion_final_y - posicion_inicial_y

    pasos_x = abs(int(delta_x / paso_um_x))
    pasos_y = abs(int(delta_y / paso_um_y))

    desp_x = np.linspace(posicion_inicial_x, posicion_final_x, pasos_x + 1)
    desp_y = np.linspace(posicion_inicial_y, posicion_final_y, pasos_y + 1)
    
    dcx, dcy = [], []
    epsilon = 1e-3 
    guess_y = max(epsilon, dcy_inicial)
    
    for i in range(len(desp_y)):
        guess_x = max(epsilon, dcx_inicial) 
        for j in range(len(desp_x)):
            nuevo_dcx, nuevo_dcy = IteracionesNR(200, guess_x, guess_y, desp_x[j], desp_y[i])
            dcx.append(nuevo_dcx)
            dcy.append(nuevo_dcy)
            guess_x = max(epsilon, nuevo_dcx)
        guess_y = max(epsilon, nuevo_dcy)

    return dcx, dcy, desp_x, desp_y

def recortar_bordes(dcx, dcy, desp_x, desp_y, n_x, n_y):
    """
    Recorta 'n_x' puntos en los bordes de X y 'n_y' puntos en los bordes de Y.
    """
    Nx = len(desp_x)  # <-- CORREGIDO: El ancho real es la cantidad de pasos en X
    Ny = len(desp_y)  # <-- CORREGIDO: El alto real es la cantidad de pasos en Y
    
    if 2 * n_x >= Nx or 2 * n_y >= Ny:
        raise ValueError('El recorte solicitado es demasiado grande para las dimensiones de la grilla.')

    # 1. Transformamos a matrices 2D usando las dimensiones correctas
    dcx_2d = np.array(dcx).reshape((Ny, Nx))
    dcy_2d = np.array(dcy).reshape((Ny, Nx))
    
    # 2. Rebanamos de forma segura
    slice_y = slice(n_y, -n_y) if n_y > 0 else slice(None)
    slice_x = slice(n_x, -n_x) if n_x > 0 else slice(None)
    
    dcx_crop = dcx_2d[slice_y, slice_x]
    dcy_crop = dcy_2d[slice_y, slice_x]
    
    desp_x_crop = desp_x[slice_x]
    desp_y_crop = desp_y[slice_y]
    
    dcx_final = dcx_crop.flatten().tolist()
    dcy_final = dcy_crop.flatten().tolist()
    
    return dcx_final, dcy_final, desp_x_crop, desp_y_crop


# ---------------------------------------------------------
# 3. EJECUCIÓN Y VALIDACIÓN (TEST)
# ---------------------------------------------------------

Paso_x = 0.05  # micrones
Paso_y = 0.75

# 1. Calculamos la grilla completa
dcx_calc, dcy_calc, obj_x, obj_y = desplazamientos(0.0, 1.0, 0.0, 1.0, paso_um_x=Paso_x, paso_um_y=Paso_y)
print(f"Puntos originales: {len(dcx_calc)} (Grilla de {len(obj_y)}x{len(obj_x)})")

# CORREGIDO: El porcentaje de recorte se calcula sobre la dimensión X, no sobre el total plano
Puntos_x_a_recortar = int(len(obj_x) * 0.1)  
Puntos_y_a_recortar = 2  

# 2. Aplicamos el recorte de los bordes
dcx_calc, dcy_calc, obj_x, obj_y = recortar_bordes(dcx_calc, dcy_calc, obj_x, obj_y, n_x=Puntos_x_a_recortar, n_y=Puntos_y_a_recortar)
print(f"Puntos después del recorte: {len(dcx_calc)} (Grilla de {len(obj_y)}x{len(obj_x)})")

# --- PRUEBA DE CINEMÁTICA DIRECTA ---
x_verificacion = []
y_verificacion = []

for idx in range(len(dcx_calc)):
    x_val = dcx_calc[idx]
    y_val = dcy_calc[idx]
    
    # CORREGIDO: Se cambió el signo '-' por '+' para que coincida con el modelo de NR
    X_fisico = polinomio_x(x_val) + cross_x(polinomio_y(y_val))
    Y_fisico = polinomio_y(y_val) + cross_y(polinomio_x(x_val))
    
    x_verificacion.append(X_fisico)
    y_verificacion.append(Y_fisico)

# Graficamos
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(dcx_calc, label='Duty Cycle X', marker='o', markersize=3, linestyle='-')
plt.plot(dcy_calc, label='Duty Cycle Y', marker='x', markersize=3, linestyle='-')
plt.title('Señales de Control (Duty Cycle)')
plt.xlabel('Paso en el escaneo (Tiempo)')
plt.ylabel('Duty Cycle [0, 1]')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x_verificacion, y_verificacion, c='red', s=10, label='Puntos medidos (calc)')
plt.title('Simulación de Trayectoria Física [um]')
plt.xlabel('Eje X [um]')
plt.ylabel('Eje Y [um]')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

dutys_csv = pd.DataFrame({'Dcx': dcx_calc,'Dcy':dcy_calc})
dutys_csv.to_csv(r'dutys_disc_v2(NR)_1906_continuo.csv', index=False)