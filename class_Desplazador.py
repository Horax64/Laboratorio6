"""
@author: tomas
"""
from machine import Pin, PWM
from time import sleep,time
from esp32 import PCNT
import Newton_Raphson as NR

max_duty = 2**16 - 1

class Desplazador:
    """
    Clase para controlar un desplazador en 3 ejes (X, Y, Z) usando una ESP32.
    Permite generar PWM en los ejes, centrar posiciones, medir con contador
    de pulsos y realizar barridos automáticos en XY.
    """

    def __init__(self, pin_x=25, pin_y=33, pin_z=32, pin_counter=27):
        """
        Inicializa el desplazador configurando los pines de PWM y el contador.

        Parámetros:
            pin_x (int): pin de salida PWM para el eje X.
            pin_y (int): pin de salida PWM para el eje Y.
            pin_z (int): pin de salida PWM para el eje Z.
            pin_counter (int): pin de entrada para el contador de pulsos.
        """
        self.pwm_x = PWM(Pin(pin_x), freq=1221)
        self.pwm_y = PWM(Pin(pin_y), freq=1221)
        self.pwm_z = PWM(Pin(pin_z), freq=1221)
        self.counter = PCNT(0, pin=Pin(pin_counter), rising=PCNT.INCREMENT)   

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
            self.pwm_y.duty_u16(dc_y)
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
            self.pwm_z.duty_u16(dc_z)
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
        cant_pasos_y = len(self.dcy)

        t_inicio_barrido = time()
        print(self.dcx)
        print(self.dcy)

        try:
            for j in range(cant_pasos_y):
                # 1. Posicionamiento en el eje Y (Fila actual)
                val_y = int(self.dcy[j] * max_duty)
                self.y(val_y)

                val_x = int(self.dcx[j] * max_duty)
                self.x(val_x)
                    
                # Estabilización nanométrica
                sleep(t_slp)
                        
        except KeyboardInterrupt:
            print("\n[WARN] Barrido abortado. Compilando datos alcanzados...")
        
        finally:
            duracion = time() - t_inicio_barrido
            print(f"Proceso finalizado en {duracion/60:.2f} minutos.")
            

    def barrido_calibracion(self,dir_barrido):
        i = 0
        if dir_barrido == "y":
            ys = range(0,65536,3855)
            xs = range(0,65536,3855)
            for dcx in xs:
                self.x(dcx)
                self.y(0)
                sleep(1)
                for dcy in ys:
                    self.y(dcy)
                    sleep(0.5)
        elif dir_barrido == "x":
            ys = range(0,65537,3855)
            xs = range(0,65537,3855)
            for dcy in ys:
                self.y(dcy)
                self.x(0)
                sleep(1)
                for dcx in xs:
                    self.x(dcx)
                    sleep(0.5)
                    
    def barrido_auxiliar(self, xlims, ylims, paso_x, paso_y, tslp_x, tslp_y, dir_barrido):
        ys = range(ylims[0],ylims[1],paso_y)
        xs = range(xlims[0],xlims[1],paso_x)
        i = 0
        if dir_barrido == "y":
            for dcx in xs:
                self.x(dcx)
                self.y(0)
                sleep(tslp_x)
                for dcy in ys:
                    self.y(dcy)
                    sleep(tslp_y)
        elif dir_barrido == "x":
            for dcy in ys:
                self.y(dcy)
                self.x(0)
                sleep(tslp_y)
                for dcx in xs:
                    self.x(dcx)
                    sleep(tslp_x)
                    
    def barrido_prueba(self):
        with open('dutys_scanning_xy_0807_1658.txt', 'r', encoding='utf-8') as archivo:
            i = 0
            for linea in archivo:
                # Limpiamos el salto de línea final y separamos por comas
                valores = linea.strip().split(',')
                dcx = max_duty*float(valores[1])
                dcy = max_duty*float(valores[2])
                if dcy < 0:
                    dcy = 0
                if dcx > max_duty:
                    dcx = max_duty
                self.x(int(dcx))
                self.y(int(dcy))
                if i != 0:
                    sleep(0.01)
                elif i == 0:
                    sleep(2)
                i += 1

