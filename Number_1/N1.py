# -*- coding: utf-8 -*-
'''
Вейвлет Рикера распространяется в одну сторону (TFSF boundary).
Область моделирования - свободное пространство.
'''
from json import tool
import numpy as np

import tools



import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return round(x / self.discrete)



class HarmonicPlaneWave():
    ''' Класс с уравнением плоской волны для гармонического сигнала в дискретном виде
    magnitude - амплитуда сигнала
    Nl - количество ячеек на длину волны.
    phi0 - начальная фаза.
    Sc - число Куранта.
    eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    mu - относительная магнитная проницаемость среды, в которой расположен источник.
    '''

    def __init__(self, magnitude, Nl, phi0, dt,  Sc=1.0, eps=1.0, mu=1.0):
        self.magnitude = magnitude
        self.Nl = Nl
        self.phi0 = phi0
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        self.dt = dt

    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q / self.Nl
        '''
        return self.magnitude * np.sin(2 * np.pi * 1.2 * 1e9 * dt  * (self.Sc * q - np.sqrt(self.mu * self.eps) * m) + self.phi0)
    

if __name__ == '__main__':
    
    c = 3e8

    # Волновое сопротивление свободного пространства
    eps_r = 2.5
    W0 = 120.0 * np.pi
    
    # Расчет длинны волны через частоты сиганал
    f_min = 0.1e9
    f_max = 2e9
    f_mid = (f_max+f_min)/2
    lamda = c / f_mid
                   
    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в отсчетах
    maxSize_m = 5
    dx = 5e-3
    dt = dx/c
    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)
    PecX = maxSize - 1

    # Положение источника в отсчетах
    sourcePos_m = 1
    sourcePos = sampler_x.sample(sourcePos_m)
    
    # Время расчета в отсчетах
    maxTime_s = 60e-9
    maxTime = sampler_t.sample(maxTime_s)
    # Параметры Вейвлета Рикера
    Np = 55.0
    Md = 2.5

    # Датчики для регистрации поля
    probepos_m = 2
    probesPos = [sampler_x.sample(probepos_m)]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    
    eps = np.ones(maxSize)
    eps[:]=eps_r
    
    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)
    

    source = HarmonicPlaneWave (1.0, Nl, phi0, dt, Sc, eps=6.0)
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    # Расчет коэффициентов для граничных условий для ABC первой степени 
    tempLeft = Sc / np.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)

    # Ez[1] в предыдущий момент времени для ABC первой степени
    oldEzLeft = Ez[1]


    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(dx, dt, maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))
        
        # Граничные условия ABC первой степени
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]
        
        #Граничные условия PEC
        Ez[PecX] = 0.0
        
        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 50 == 0:
            display.updateData(display_field, q)
    display.stop()
    
    tools.showProbeSpectrum(probes, dx, dt, -5, 5)