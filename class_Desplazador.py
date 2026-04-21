"""
@author: horax & fedex
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

    def __init__(self, pin_x=33, pin_y=32, pcentrar_z=25, pin_counter=27):

        """
        Inicializa el desplazador configurando los pines de PWM y el contador.

        Parámetros:
            pin_x (int): pin de salida PWM para el eje X.
            pin_y (int): pin de salida PWM para el eje Y.
            pcentrar_z (int): pin de salida PWM para el eje Z.
            pin_counter (int): pin de entrada para el contador de pulsos.
        """
        self.pwm_x = PWM(Pin(pin_x), freq=1221)
        self.pwm_y = PWM(Pin(pin_y), freq=1221)
        self.pwm_z = PWM(Pin(pcentrar_z), freq=1221)
        self.counter = PCNT(0, pin=Pin(pin_counter), rising=PCNT.INCREMENT)   

    def configurar_dc_x(self, dc_x):
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
            
    def configurar_dc_y(self, dc_y):
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
            
    def configurar_dc_z(self, dc_z):
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
        
    def centro(self, centrar_z=True):
        """
        Centra el desplazador en los ejes X e Y.
        Si centrar_z=False, también centra el eje Z.

        Parámetros:
            centrar_z (bool): si es True, centra los 3 ejes; por defecto solo X e Y.
        """
        if centrar_z:
            self.configurar_dc_x(.5)
            self.configurar_dc_y(.5)
            self.configurar_dc_z(.5)
        else:
            self.configurar_dc_x(.5)
            self.configurar_dc_y(.5)


    def medir_pulsos(self, t_int, count):

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
        Realiza un barrido automático en los ejes X e Y usando desplazamientos
        calculados con Newton_Raphson.desplazamientos(). Comienza desde la esquina
        inferior izquierda. Para (50%, 50%) corresponden, aproximadamente, los
        valores (4 um, -3.3 um) de desplazamiento. 

        Parámetros:
            param (tuple): self.dcx_inicial, self.dcy_inicial, self.dcx_final, self.dcy_final, y paso (um).
            t_int (float): tiempo de integración para cada medición.
            t_slp (float): tiempo de espera entre pasos.
            count (list) : lista donde se almacenan las mediciones.
        """
        print('Calculando los duty cycles...')
        
        inicio = time()
        self.dcx, self.dcy, cant_pasos_x = NR.desplazamientos(*param)
        print(f'La cantidad de pasos es {cant_pasos_x}')
        fin = time()
        
        print(f'Listo! Tardo {fin-inicio} s')
            
        self.configurar_dc_y(int(self.dcy[0]*max_duty))
        print(int(self.dcy[0]*max_duty))
        print(len(self.dcx))
        print(self.dcx)
        
        for i in range(0,len(self.dcx)):
            #print(self.dcx[i]*max_duty)
            self.configurar_dc_x(int(self.dcx[i]*max_duty))
            sleep(t_slp)

    '''
    Código nuevo de acá para abajo
    '''


    def barrido_auxiliar(self, xlims, ylims, tslp):
        xs = range(xlims[0],xlims[1],100)
        ys = range(ylims[0],ylims[1],5)
        for dcy in ys:
            self.configurar_dc_x(0)
            self.configurar_dc_y(dcy)
            sleep(tslp)
            for dcx in xs:
                self.configurar_dc_x(dcx)
                sleep(tslp)
        
