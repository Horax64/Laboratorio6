import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Desactivar notación científica en numpy para leer más fácil
np.set_printoptions(suppress=True)

# ---------------------------------------------------------
# 1. CARGA Y DEFINICIÓN DEL MODELO DIRECTO
# ---------------------------------------------------------

promedio_cross_x = [-0.03461811951503092, 0]
promedio_cross_y = [-0.07531236878047691, 0]

try:
    # Intenta leer tus datos de ajuste. 
    # Al usar poly1d, si las columnas de promedio_x tienen 4 elementos, arma grado 3 solo.
    datos_ajuste_x = pd.read_csv('ajuste_cubico_x.csv') 
    datos_ajuste_y = pd.read_csv('ajuste_cubico_y.csv')

    promedio_x = datos_ajuste_x.mean()[1::]
    coefs_x = [coef for coef in promedio_x] 
    coefs_x[-1] = 0

    promedio_y = datos_ajuste_y.mean()[1::]
    coefs_y = [coef for coef in promedio_y] 
    coefs_y[-1] = 0
except FileNotFoundError:
    print("No se encontraron los CSV. Usando coeficientes cúbicos de prueba...")
    # Coeficientes dummy si no tenés los CSV a mano (para que el código corra igual)
    coefs_x = [2.0, -0.5, 4.0, 0.0]  # Px(x)
    coefs_y = [1.8, -0.4, 3.8, 0.0]  # Py(y)

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

def desplazamientos(dcx_inicial, dcx_final, dcy_inicial, dcy_final, paso_um):
    posicion_inicial_x = polinomio_x(dcx_inicial) + cross_x(polinomio_y(dcy_inicial))
    posicion_inicial_y = polinomio_y(dcy_inicial) + cross_y(polinomio_x(dcx_inicial))
    posicion_final_x   = polinomio_x(dcx_final) + cross_x(polinomio_y(dcy_final))
    posicion_final_y   = polinomio_y(dcy_final) + cross_y(polinomio_x(dcx_final))

    delta_x = posicion_final_x - posicion_inicial_x
    delta_y = posicion_final_y - posicion_inicial_y

    pasos_x = abs(int(delta_x / paso_um))
    pasos_y = abs(int(delta_y / paso_um))

    # CLAVE: np.linspace maneja perfecto si el barrido tiene que ir hacia "atrás" o hacia "adelante"
    desp_x = np.linspace(posicion_inicial_x, posicion_final_x, pasos_x + 1)
    desp_y = np.linspace(posicion_inicial_y, posicion_final_y, pasos_y + 1)
    
    dcx, dcy = [], []
    
    epsilon = 1e-3 
    guess_y = max(epsilon, dcy_inicial)
    
    for i in range(len(desp_y)):
        guess_x = max(epsilon, dcx_inicial) 
        
        for j in range(len(desp_x)):
            # Le damos 200 iteraciones para que tenga tiempo de caminar de a pasitos cortos de 0.05
            nuevo_dcx, nuevo_dcy = IteracionesNR(200, guess_x, guess_y, desp_x[j], desp_y[i])
            
            dcx.append(nuevo_dcx)
            dcy.append(nuevo_dcy)
            
            guess_x = max(epsilon, nuevo_dcx)
            
        guess_y = max(epsilon, nuevo_dcy)

    return dcx, dcy, desp_x, desp_y


# ---------------------------------------------------------
# 3. EJECUCIÓN Y VALIDACIÓN (TEST)
# ---------------------------------------------------------

# ATENCIÓN ACÁ HORAX: El "paso" ahora debe estar en micrómetros. 
# Si tu rango físico es de aprox 5 um, un paso de 0.5 um te da 10 puntos por lado.
PASO_MICRONES = 0.5 

dcx_calc, dcy_calc, obj_x, obj_y = desplazamientos(0.0, 1.0, 0.0, 1.0, paso_um=PASO_MICRONES)

print(f"Puntos totales calculados: {len(dcx_calc)}")

# --- PRUEBA DE CINEMÁTICA DIRECTA ---
# Metemos los duty cycles calculados en las ecuaciones originales
# para ver si físicamente el desplazador va a hacer la grilla perfecta.

x_verificacion = []
y_verificacion = []

for idx in range(len(dcx_calc)):
    x_val = dcx_calc[idx]
    y_val = dcy_calc[idx]
    
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
plt.plot(x_verificacion, y_verificacion, c='blue', alpha=0.3, linewidth=1, label='Trayectoria')
plt.title('Simulación de Trayectoria Física [um]')
plt.xlabel('Eje X [um]')
plt.ylabel('Eje Y [um]')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()