import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# Desactivar notación científica en numpy para leer más fácil
np.set_printoptions(suppress=True)

# ---------------------------------------------------------
# 1. CARGA Y DEFINICIÓN DEL MODELO DIRECTO
# ---------------------------------------------------------

promedio_cross_x = pd.read_csv(r'Calibración\Datos_ajuste\ajuste_lin_x_calv1_3006.csv')
promedio_cross_y = pd.read_csv(r'Calibración\Datos_ajuste\ajuste_lin_y_calv1_3006.csv')
promedio_cross_x = promedio_cross_x[5:] #Los primeros datos son outliers

promedio_cross_x = [promedio_cross_x['m'].mean(), 0]
promedio_cross_y = [promedio_cross_y['m'], 0]

# Intenta leer tus datos de ajuste. 
# Al usar poly1d, si las columnas de promedio_x tienen 4 elementos, arma grado 3 solo.
datos_ajuste_x = pd.read_csv(r'Calibración\Datos_ajuste\ajuste_cubico_x_calv1_3006.csv') 
datos_ajuste_y = pd.read_csv(r'Calibración\Datos_ajuste\ajuste_cubico_y_calv1_3006.csv')

promedio_x = datos_ajuste_x.mean()[1::]
coefs_x = [coef for coef in promedio_x] 
coefs_x[-1] = 0

promedio_y = datos_ajuste_y.mean()[1::]
coefs_y = [coef for coef in promedio_y] 
coefs_y[-1] = 0


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
    
    # Tu cambio: perfecto
    F_x = Px + cross_x(Py) - X
    F_y = Py + cross_y(Px) - Y

    dPx = d_polinomio_x(dc_x_guess)
    dPy = d_polinomio_y(dc_y_guess)

    dFxdx = dPx
    # ACÁ ESTABA EL ERROR: Faltaba el signo menos para que coincida con F_x
    dFxdy = promedio_cross_x[0] * dPy       
    dFydx = promedio_cross_y[0] * dPx       
    dFydy = dPy

    det_J = dFxdx * dFydy - dFxdy * dFydx
    
    if abs(det_J) < 1e-12:
        return dc_x_guess, dc_y_guess
    
    # Calculamos el incremento (el paso) puro dictado por la derivada
    paso_x = (F_x * dFydy - F_y * dFxdy) / det_J
    paso_y = (-F_x * dFydx + F_y * dFxdx) / det_J

    # CLAVE: Damped Newton. Limitamos el tamaño máximo del paso para evitar rebotes
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
         
         # Clipping entre 0 y 1
         x_new = max(0.0, min(1.0, x_new))
         y_new = max(0.0, min(1.0, y_new))
         
         if abs(x_new - x) < tol and abs(y_new - y) < tol:
             return x_new, y_new
             
         x, y = x_new, y_new
         
    return x, y

def desplazamientos(dcx_inicial, dcy_inicial, dcx_final, dcy_final, paso_x, paso_y):
    
    # 1. Cinemática Directa
    inicio_x = polinomio_x(dcx_inicial) + cross_x(polinomio_y(dcy_inicial))
    inicio_y = polinomio_y(dcy_inicial) + cross_y(polinomio_x(dcx_inicial))
    fin_x = polinomio_x(dcx_final) + cross_x(polinomio_y(dcy_final))
    fin_y = polinomio_y(dcy_final) + cross_y(polinomio_x(dcx_final))

    
    # Imprimimos los límites físicos para debuguear y que veas por qué daba vacío
    print(f"Limites X [um]: {inicio_x:.3f} a {fin_x:.3f} (Paso: {paso_x})")
    print(f"Limites Y [um]: {inicio_y:.3f} a {fin_y:.3f} (Paso: {paso_y})")
    
    # 2. Generación robusta de vectores (con abs() para soportar barridos inversos)
    pasos_totales_x = abs(int((fin_x - inicio_x) / paso_x))
    pasos_totales_y = abs(int((fin_y - inicio_y) / paso_y))
    
    # Definimos el signo del paso (si va de mayor a menor, el paso es negativo)
    signo_x = 1 if fin_x >= inicio_x else -1
    signo_y = 1 if fin_y >= inicio_y else -1
    
    desp_x = [round(inicio_x + i * paso_x * signo_x, 3) for i in range(pasos_totales_x + 1)]
    desp_y = [round(inicio_y + j * paso_y * signo_y, 3) for j in range(pasos_totales_y + 1)]

    dcx, dcy = [], []
    epsilon = 1e-3 
    guess_y = max(epsilon, dcy_inicial)
    
    for i in range(len(desp_y)):
        guess_x = max(epsilon, dcx_inicial) 
        
        # Inicializamos nuevo_dcy con el guess actual por si desp_x está vacío
        nuevo_dcy = guess_y 
        
        for j in range(len(desp_x)):
            nuevo_dcx, nuevo_dcy = IteracionesNR(200, guess_x, guess_y, desp_x[j], desp_y[i])
            dcx.append(nuevo_dcx)
            dcy.append(nuevo_dcy)
            guess_x = max(epsilon, nuevo_dcx)
            
        guess_y = max(epsilon, nuevo_dcy)

    return dcx, dcy, desp_x, desp_y

def recortar_bordes(dcx, dcy, desp_x, desp_y, n,m):
    """
    Recorta 'n' puntos de los 4 bordes de la grilla de escaneo.
    """
    Nx = len(desp_x)
    Ny = len(desp_y)
    
    # Chequeo de seguridad: evitar recortar más puntos de los que existen
    if 2*n >= Nx or 2*m >= Ny:
        print("El margen es muy grande para esta grilla. Te quedás sin puntos.")
        return dcx, dcy, desp_x, desp_y
        
    # 1. Transformamos las listas 1D en matrices 2D (Ny filas, Nx columnas)
    dcx_2d = np.array(dcx).reshape((Ny, Nx))
    dcy_2d = np.array(dcy).reshape((Ny, Nx))
    
    # 2. Rebanamos la matriz: sacamos 'n' filas y 'n' columnas de cada extremo
    dcx_crop = dcx_2d[m:-m, n:-n]
    dcy_crop = dcy_2d[m:-m, n:-n]
    
    # 3. Recortamos también los vectores físicos de referencia
    desp_x_crop = desp_x[n:-n]
    desp_y_crop = desp_y[m:-m]
    
    # 4. Volvemos a aplanar la matriz a una lista 1D para que el hardware la consuma
    dcx_final = dcx_crop.flatten().tolist()
    dcy_final = dcy_crop.flatten().tolist()
    
    return dcx_final, dcy_final, desp_x_crop, desp_y_crop


# ---------------------------------------------------------
# 3. EJECUCIÓN Y VALIDACIÓN (TEST)
# ---------------------------------------------------------

# ATENCIÓN ACÁ HORAX: El "paso" ahora debe estar en micrómetros. 
# Si tu rango físico es de aprox 5 um, un paso de 0.5 um te da 10 puntos por lado.
# ---------------------------------------------------------
# 3. EJECUCIÓN Y VALIDACIÓN (TEST)
# ---------------------------------------------------------

PASO_MICRONES_X = 0.5
PASO_MICRONES_Y = 0.5
PUNTOS_A_RECORTAR_X = 3
PUNTOS_A_RECORTAR_Y = 2  # Acá definís cuántos puntos volás de cada borde


# 1. Calculamos la grilla completa
dcx_calc, dcy_calc, obj_x, obj_y = desplazamientos(0.0, 0.0, 1.0, 1.0, paso_x=PASO_MICRONES_X,paso_y=PASO_MICRONES_Y)
print(f"Puntos originales: {len(dcx_calc)}")

# 2. Aplicamos el recorte de los bordes
dcx_calc, dcy_calc, obj_x, obj_y = recortar_bordes(dcx_calc, dcy_calc, obj_x, obj_y, n=PUNTOS_A_RECORTAR_X,m=PUNTOS_A_RECORTAR_Y)
print(f"Puntos después del recorte: {len(dcx_calc)}")

# --- PRUEBA DE CINEMÁTICA DIRECTA ---
# Metemos los duty cycles calculados en las ecuaciones originales
# para ver si físicamente el desplazador va a hacer la grilla perfecta.

# --- PRUEBA DE CINEMÁTICA DIRECTA ---
x_verificacion = []
y_verificacion = []

for idx in range(len(dcx_calc)):
    x_val = dcx_calc[idx]
    y_val = dcy_calc[idx]
    
    # ACÁ: Mismo cambio, respetar el modelo físico
    X_fisico = polinomio_x(x_val) + cross_x(polinomio_y(y_val))
    Y_fisico = polinomio_y(y_val) + cross_y(polinomio_x(x_val))
    
    x_verificacion.append(X_fisico)
    y_verificacion.append(Y_fisico)

# Graficamos
plt.figure(figsize=(10, 5))

# Gráfico 1: Los Duty Cycles eléctricos (cómo se ven las señales)
plt.subplot(1, 2, 1)
plt.plot(dcx_calc, label='Duty Cycle X', marker='o', markersize=3, linestyle='-')
plt.plot(dcy_calc, label='Duty Cycle Y', marker='x', markersize=3, linestyle='-')
plt.title('Señales de Control (Duty Cycle)')
plt.xlabel('Paso en el escaneo (Tiempo)')
plt.ylabel('Duty Cycle [0, 1]')
plt.grid(True, alpha=0.3)
plt.legend()

# Gráfico 2: Trayectoria Física Real (Cinemática Directa)
plt.subplot(1, 2, 2)
plt.scatter(x_verificacion, y_verificacion, c='red', s=10, label='Puntos medidos (calc)')
#plt.plot(x_verificacion, y_verificacion, c='blue', alpha=0.3, linewidth=1, label='Trayectoria')
plt.title('Simulación de Trayectoria Física [um]')
plt.xlabel('Eje X [um]')
plt.ylabel('Eje Y [um]')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

dutys_csv = pd.DataFrame({'Dcx': dcx_calc,'Dcy':dcy_calc})
hora = time.strftime("%d%m_%H%M")
print(type(hora))

dutys_csv.to_csv(fr'dutys_barrido_discreto_x_{hora}.csv')

