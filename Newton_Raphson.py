"""
@author: horax & fedex
"""

# Coeficientes encontrados a partir de ajustar las curvas de histérisis en x e y

coeficientes_ajustados_pol4_x      = [-2.63871026e-09, -1.52030874e-06,  4.80642439e-04,  6.35644438e-02, -2.38560953e-02]
coeficientes_cross_en_x            = [-0.00301124,  0.00139561]
coeficientes_ajustados_pol4_y      = [ 1.71210520e-08, -1.89181021e-06, -2.07786017e-04, -5.68752641e-02, 3.16507343e-02]
coeficientes_cross_en_y            = [ 0.00475161, -0.06391525]

# Modelo para la curva de histérisis en sentido ascendente de duty-cycle, para cada eje
def pol4_x(dc):
    coef_x = coeficientes_ajustados_pol4_x
    return coef_x[0]*(x**4) + coef_x[1]*(x**3) + coef_x[2]*(x**2) + coef_x[3]*x + coef_x[4] 

def pol4_y(dc):
    coef_y = coeficientes_ajustados_pol4_y
    return coef_y[0]*(dc**4) + coef_y[1]*(dc**3) + coef_y[2]*(dc**2) + coef_y[3]*dc + coef_y[4]

# Modelo para el cross-talk entre los ejes x e y
def lineal_x(dc):
    coef_cross_x = coeficientes_cross_en_x
    return coef_cross_x[0]*dc + coef_cross_x[1]

def lineal_y(dc):
    coef_cross_y = coeficientes_cross_en_y
    return coef_cross_y[0]*dc + coef_cross_y[1]

# Derivada del polinomio de grado 4
def derivada_pol_4_x(x):
    coef = coeficientes_ajustados_pol4_x[:-1]
    return 4*coef[0]*(x**3) + 3*coef[1]*(x**2) + 2*coef[2]*x + coef[3]

def derivada_pol_4_y(x):
    coef = coeficientes_ajustados_pol4_y[:-1]
    return 4*coef[0]*(x**3) + 3*coef[1]*(x**2) + 2*coef[2]*x + coef[3]


# Newton-Raphson para un paso
def NewtonRaphsonIndividual(dc_x_guess, dc_y_guess, X, Y):
    
    """
    Si las posiciones físicas que se quieren setear son (X,Y) y los initial guesses para los 
    duty-cycles correspondientes son (dc_x_guess,dc_y_guess), pedimos que:

    * pol4(x0) + lin(x0) = X
    * pol4(y0) + lin(y0) = y

    Queremos encontrar los valores de duty-cycle que mejor ajusten estas ecuaciones. Podemos reducir este 
    problema a intentar IteracionesNRimar los ceros de las funciones mediante el método de Newton-Raphson:

    * F_x(x0) = pol4(x0) + lin(x0) - X
    * F_y(x0) = pol4(y0) + lin(y0) - Y

    Esta función se encarga de realizar una iteración con el método.
    """


    # Funciones F_x y F_y

    F_x = pol4_x(dc_x_guess) + lineal_x(dc_y_guess) - X
    F_y = pol4_y(dc_y_guess) + lineal_y(dc_x_guess) - Y


    # Elementos de la matriz jacobiana J

    dFxdx = derivada_pol_4_x(dc_x_guess)
    dFxdy = coeficientes_cross_en_x[0]       # f_y
    dFydx = coeficientes_cross_en_y[0]       # f_x
    dFydy = derivada_pol_4_y(dc_y_guess)

    # determinante de J
    det_J = dFxdx*dFydy - dFxdy*dFydx

    # Iteración de Newton-Raphson
    dc_x_obtenido = dc_x_guess - (F_x*dFydy - F_y*dFxdy) / det_J
    dc_y_obtenido = dc_y_guess - (-F_x*dFydx + F_y*dFxdx) / det_J

    return dc_x_obtenido, dc_y_obtenido

# Iteración múltiple de NewtonRaphsonIndividual

def IteracionesNR(n, dc_x_guess, dc_y_guess, X, Y):

    """
    Usando la función NewtonRaphsonIndividual hacemos n iteraciones con el método
    a partir de un guess inicial (x0,y0). 
    """
    x, y = NewtonRaphsonIndividual(dc_x_guess, dc_y_guess, X, Y)

    for _ in range(n):
         x, y = NewtonRaphsonIndividual(x, y, X, Y)
    return x, y

def desplazamientos(dcx_inicial, dcy_inicial, dcx_final, dcy_final, paso):

    max_duty = 2**16 - 1

    dcx_inicial = dcx_inicial*100/max_duty
    dcy_inicial = dcy_inicial*100/max_duty
    dcx_final = dcx_final*100/max_duty
    dcy_final = dcy_final*100/max_duty

    posicion_inicial_x =  pol4_x(dcx_inicial) + lineal_x(dcy_inicial)
    posicion_inicial_y =  pol4_y(dcy_inicial) + lineal_y(dcx_inicial)
    posicion_final_x   =  pol4_x(dcx_final) + lineal_x(dcy_final)
    posicion_final_y   =  pol4_y(dcy_final) + lineal_y(dcx_final)

    # Arrays de desplazamientos deseado
    desp_x = [round(posicion_inicial_x + i*paso, 3) for i in range(int((posicion_final_x - posicion_inicial_x)/paso)+1)]
    desp_y = [round(posicion_inicial_y - i*paso, 3) for i in range(abs(int((posicion_final_y - posicion_inicial_y)/paso))+1)]
    
    cant_pasos_x = int((posicion_final_x - posicion_inicial_x)/paso)+1 

    # Arrays de duty-cycles

    dcx = []
    dcy = []
    
    for i in range(len(desp_y)):
        for j in range(len(desp_x)):
            if j==0:     
                if i ==0:
                    dcx_i, dcy_i = IteracionesNR(20,dcx_inicial, dcy_inicial, desp_x[j], desp_y[i])
                else:
                    dcx_i, dcy_i = IteracionesNR(20,dcx_inicial, dcy[-1], desp_x[j], desp_y[i])
            else:
                dcx_i, dcy_i = IteracionesNR(20,dcx[-1], dcy[-1], desp_x[j], desp_y[i])
            
            dcx.append(dcx_i)
            dcy.append(dcy_i)
            
    dcx = [round(i/100, 3) for i in dcx]
    dcy = [round(i/100, 3) for i in dcy]

    return dcx, dcy, cant_pasos_x

