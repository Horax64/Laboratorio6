hist_x      = [-2.63871026e-09, -1.52030874e-06,  4.80642439e-04,  6.35644438e-02, -2.38560953e-02]
cross_en_x  = [-0.00301124,  0.00139561]
hist_y      = [ 1.71210520e-08, -1.89181021e-06, -2.07786017e-04, -5.68752641e-02, 3.16507343e-02]
cross_en_y  = [ 0.00475161, -0.06391525]

# Newton-Raphson para un paso
def NR(x0, y0, X, Y):
    
    # Funciones F_x y F_y
    Fx = pol4(x0, *hist_x) + lin(y0, *cross_en_x) - X
    Fy = pol4(y0, *hist_y) + lin(x0, *cross_en_y) - Y
    # Jacobiano
    dFxdx = d_pol4(x0, *hist_x[:-1])
    dFxdy = cross_en_x[0]       # f_y
    dFydx = cross_en_y[0]       # f_x
    dFydy = d_pol4(y0, *hist_y[:-1])
    # determinante de J
    det_J = dFxdx*dFydy - dFxdy*dFydx
    # Actualización Newton-Raphson
    x1 = x0 - (Fx*dFydy - Fy*dFxdy) / det_J
    y1 = y0 - (-Fx*dFydx + Fy*dFxdx) / det_J
    return x1, y1

# Iteración múltiple de NR
def aprox(x0, y0, X, Y):
    x, y = NR(x0, y0, X, Y)
    for _ in range(20):
         x, y = NR(x, y, X, Y)
    return x, y

def desplazamientos(dcx_inicial, dcy_inicial, dcx_final, dcy_final, paso):
    max_duty = 2**16 - 1
    dcx_inicial = dcx_inicial*100/max_duty
    dcy_inicial = dcy_inicial*100/max_duty
    dcx_final = dcx_final*100/max_duty
    dcy_final = dcy_final*100/max_duty

    inicio_x =  pol4(dcx_inicial, *hist_x) + lin(dcy_inicial, *cross_en_x)
    inicio_y =  pol4(dcy_inicial, *hist_y) + lin(dcx_inicial, *cross_en_y)
    fin_x =  pol4(dcx_final, *hist_x) + lin(dcy_final, *cross_en_x)
    fin_y =  pol4(dcy_final, *hist_y) + lin(dcx_final, *cross_en_y)

    # Arrays de desplazamientos deseado
    desp_x = [round(inicio_x + i*paso, 3) for i in range(int((fin_x - inicio_x)/paso)+1)]
    desp_y = [round(inicio_y - i*paso, 3) for i in range(abs(int((fin_y - inicio_y)/paso))+1)]
    
    cant_pasos_x = int((fin_x - inicio_x)/paso)+1 
    # Resolver 
    dcx = []
    dcy = []
    
    for i in range(len(desp_y)):
        for j in range(len(desp_x)):
            if j==0:     
                if i ==0:
                    dcx_i, dcy_i = aprox(dcx_inicial, dcy_inicial, desp_x[j], desp_y[i])
                else:
                    dcx_i, dcy_i = aprox(dcx_inicial, dcy[-1], desp_x[j], desp_y[i])
            else:
                dcx_i, dcy_i = aprox(dcx[-1], dcy[-1], desp_x[j], desp_y[i])
            
            dcx.append(dcx_i)
            dcy.append(dcy_i)
            
    dcx = [round(i/100, 3) for i in dcx]
    dcy = [round(i/100, 3) for i in dcy]

    return dcx, dcy, cant_pasos_x

