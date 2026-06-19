"""
@author: tomas
"""
from time import sleep,time
import Newton_Raphson as NR

max_duty = 2**16 - 1

class Desplazador:
    """
    Clase para controlar un desplazador en 3 ejes (X, Y, Z) usando una ESP32.
    Permite generar PWM en los ejes, centrar posiciones, medir con contador
    de pulsos y realizar barridos automáticos en XY.
    """

    def __init__(self, pin_x=33, pin_y=32, pin_z=25, pin_counter=27):
        """
        Inicializa el desplazador configurando los pines de PWM y el contador.

        Parámetros:
            pin_x (int): pin de salida PWM para el eje X.
            pin_y (int): pin de salida PWM para el eje Y.
            pin_z (int): pin de salida PWM para el eje Z.
            pin_counter (int): pin de entrada para el contador de pulsos.
        """

    def x(self, dc_x):
        """
        Configura el duty cycle del eje X.

        Parámetros:
            dc_x (int): valor del duty cycle a usar. 
            """
        if dc_x < 0:
            print('El valor del duty cycle debe ser positivo')
        if dc_x <= max_duty:
            self.pwm_x.duty_u16(dc_x)
        else:
            print('El valor del duty cycle solicitado es mayor al máximo permitido')
            
    def y(self, dc_y):
        """
        Configura el duty cycle del eje Y.

        Parámetros:
            dc_y (int): valor del duty cycle a usar. 
            """
        if dc_y < 0:
            print('El valor del duty cycle debe ser positivo')
        if dc_y <= max_duty:
            self.pwm_x.duty_u16(dc_y)
        else:
            print('El valor del duty cycle solicitado es mayor al máximo permitido')
            
    def z(self, dc_z):
        """
        Configura el duty cycle del eje Z.

        Parámetros:
            dc_z (int): valor del duty cycle a usar. 
            """
        if dc_z < 0:
            print('El valor del duty cycle debe ser positivo')
        if dc_z <= max_duty:
            self.pwm_x.duty_u16(dc_z)
        else:
            print('El valor del duty cycle solicitado es mayor al máximo permitido')
        
    def centro(self, in_z=True):
        """
        Centra el desplazador en los ejes X e Y.
        Si in_z=False, también centra el eje Z.

        Parámetros:
            in_z (bool): si es True, centra los 3 ejes; por defecto solo X e Y.
        """
        if in_z:
            self.x(.5)
            self.y(.5)
            self.z(.5)
        else:
            self.x(.5)
            self.y(.5)


    def medir(self, t_int, count):
        """
        Realiza una medición con el contador de pulsos.

        Parámetros:
            t_int (float): tiempo de integración en segundos.
            count (list): lista donde se agrega el valor medido.

        Nota:
            El contador se reinicia después de la medición.
        """
        sleep(.1)
        self.counter.start()
        sleep(t_int)
        count.append(self.counter.value())
        self.counter.stop()
        self.counter.value(0)
 
    def barrer(self, param, t_int, t_slp):
        """
        Realiza un barrido raster unidireccional.
        Inicia cada fila siempre desde el lateral derecho para mitigar 
        la histéresis y asegurar repetibilidad espacial.
        """
        print('Iniciando configuración de barrido unidireccional...')
        
        # NR.desplazamientos devuelve los Duty Cycles calculados por Newton-Raphson
        self.dcx, self.dcy, cant_pasos_x = NR.desplazamientos(*param)
        print(self.dcx)
        print(self.dcy)
        cant_pasos_y = len(self.dcy)

        mediciones = []
        t_inicio_barrido = time()

        try:
            for j in range(cant_pasos_y):
                # 1. Posicionamiento en el eje Y (Fila actual)
                val_y = int(self.dcy[j] * max_duty)
                
                # 2. Retorno rápido al lateral derecho (Home de la fila)
                # Aplicamos un tiempo de espera mayor aquí porque el salto es largo
                val_x_inicio = int(self.dcx[0] * max_duty)
                
                print(f"Fila {j+1}/{cant_pasos_y} - Reposicionando en X_derecha...")
                sleep(t_slp * 3) # Tiempo extra para amortiguar la inercia del salto largo

                # 3. Barrido de la fila (Derecha a Izquierda)
                for i, dc_val in enumerate(self.dcx):
                    val_x = int(dc_val * max_duty)
                    
                    # Estabilización nanométrica
                    sleep(t_slp)
                    
        except KeyboardInterrupt:
            print("\n[WARN] Barrido abortado. Compilando datos alcanzados...")
        
        finally:
            duracion = time() - t_inicio_barrido
            print(f"Proceso finalizado en {duracion/60:.2f} minutos.")
            

    def barrido_auxiliar(self, xlims, ylims, tslp):
        xs = range(xlims[0],xlims[1],3276)
        i = 0
        for i in range(0,20):
            for dcx in xs:
                self.x(dcx)
                sleep(tslp)
            i+=1

d = Desplazador()
d.barrer([0,0,65535,65535,1],0,2)