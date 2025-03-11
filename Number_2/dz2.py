# -*- coding: utf-8 -*- 

#Моделирование отражения гармонического сигнала от слоя диэлектрика


import math

import numpy as np
import numpy.typing as npt

from objects import LayerContinuous, LayerDiscrete, Probe

import boundary
import sources
import tools



class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return round(x / self.discrete)


def sampleLayer(layer_cont: LayerContinuous, sampler: Sampler) -> LayerDiscrete:
    start_discrete = sampler.sample(layer_cont.xmin)
    end_discrete = (sampler.sample(layer_cont.xmax)
                    if layer_cont.xmax is not None
                    else None)
    return LayerDiscrete(start_discrete, end_discrete,
                         layer_cont.eps, layer_cont.mu, layer_cont.sigma)


def fillMedium(layer: LayerDiscrete,
               eps: npt.NDArray[np.float64],
               mu: npt.NDArray[np.float64],
               sigma: npt.NDArray[np.float64]):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
        sigma[layer.xmin: layer.xmax] = layer.sigma
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu
        sigma[layer.xmin:] = layer.sigma


if __name__ == '__main__':

    # Используемые константы
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Скорость света в вакууме
    c = 299792458.0

    # Электрическая постоянная
    eps0 = 8.854187817e-12

    # Параметры моделирования
    # Частота сигнала, Гц
    f_min = 5e9
    f_max = 30e9
    f_mid = (f_min+f_max)/2

    # Дискрет по пространству в м
    dx = 1*1e-5
    
    wavelength = c / f_mid
    Nl = wavelength / dx

    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в м
    maxSize_m = 0.22

    # Время расчета в секундах 
    maxTime_s = 2e-9

    # Положение источника в м
    sourcePos_m = 0.07

    # Параметры слоев
    d0 = 0.12
    
    eps1 = 3.5
    d1 = 0.01
    
    eps2 = 4.8
    d2 = 0.03
    
    eps3 = 6.5
    #d3 = 0.0
    
    layers_cont = [LayerContinuous(d0, d0+d1,  eps=eps1, sigma=0.0),
                   LayerContinuous(d0+d1, d0+d1+d2 , eps=eps2, sigma=0.0),
                   LayerContinuous(d0+d1+d2, eps=eps3, sigma=0.0)
                   ]

    # Скорость обновления графика поля
    speed_refresh = 500

    # Переход к дискретным отсчетам
    # Дискрет по времени
    dt = dx * Sc / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в отсчетах
    sourcePos = sampler_x.sample(sourcePos_m)

    layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]
    
    # для построения чисто падающего сигнала возьмем дополнительный датчик №2
    # Координаты датчиков для регистрации поля
    coord_probe1_m = 0.04
    coord_probe2_m = 0.08
    probesPos_m = [coord_probe1_m, coord_probe2_m]
    # Датчики для регистрации поля
    probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
    probes = [Probe(pos, maxTime) for pos in probesPos]

    posSP = 0.08
    signal_probe = Probe(sampler_x.sample(posSP), maxTime)

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = np.ones(maxSize)

    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    # Проводимость
    sigma = np.zeros(maxSize)

    for layer in layers:
        fillMedium(layer, eps, mu, sigma)

    # Источник
    aplitude = 1.0
    source = sources.ModGauss(5000, 2500, Nl, Sc, eps[sourcePos], mu[sourcePos])

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    ## Создание экземпляров классов граничных условий
    boundary_left = boundary.ABCSecondLeft(eps[0], mu[0], Sc)
    boundary_right = boundary.ABCSecondRight(eps[-1], mu[-1], Sc)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, B/m'
    display_ymin = -2.1
    display_ymax = 2.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(dx, dt,
                                        maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,
                                        title='fdtd_dielectric')

    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)
    for layer in layers:
        display.drawBoundary(layer.xmin)
        if layer.xmax is not None:
            display.drawBoundary(layer.xmax)

    for t in range(1, maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, t)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, t + 0.5))

        boundary_left.updateField(Ez, Hy)
        boundary_right.updateField(Ez, Hy)

        if(t < 10000):
            signal_probe.addData(Ez,Hy)
            
        
        ## Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в пробнике
    
    tools.showProbeFalRefSignalMy(signal_probe, probes[0], dx, dt, -2.1, 2.1)
    
    tools.showProbeFalRefSpectrumMy(signal_probe, probes[0], dx, dt, -2.1, 2.1)

    tools.showProbeKoeReflectedOfFrequency(signal_probe, probes[0], dx, dt, -2.1, 2.1)