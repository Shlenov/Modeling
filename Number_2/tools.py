# -*- coding: utf-8 -*-
'''
Модуль со вспомогательными классами и функциями, не связанные напрямую с
методом FDTD
'''

from math import gamma
import pylab
import numpy as np
import numpy.typing as npt
from typing import List, Optional

from objects import Probe


class AnimateFieldDisplay:
    '''
    Класс для отображения анимации распространения ЭМ волны в пространстве
    '''

    def __init__(self,
                 dx: float,
                 dt: float,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str,
                 title: Optional[str] = None
                 ):
        '''
        dx - дискрет по простарнству, м
        dt - дискрет по времени, сек
        maxXSize - размер области моделирования в отсчетах.
        minYSize, maxYSize - интервал отображения графика по оси Y.
        yLabel - метка для оси Y
        '''
        self.maxXSize = maxXSize
        self.minYSize = minYSize
        self.maxYSize = maxYSize
        self._xList = None
        self._line = None
        self._xlabel = 'x, м'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'
        self._dx = dx
        self._dt = dt
        self._title = title

    def activate(self):
        '''
        Инициализировать окно с анимацией
        '''
        self._xList = np.arange(self.maxXSize) * self._dx

        # Включить интерактивный режим для анимации
        pylab.ion()

        # Создание окна для графика
        self._fig, self._ax = pylab.subplots(
            figsize=(10, 6.5))

        if self._title is not None:
            self._fig.suptitle(self._title)

        # Установка отображаемых интервалов по осям
        self._ax.set_xlim(0, self.maxXSize * self._dx)
        self._ax.set_ylim(self.minYSize, self.maxYSize)

        # Установка меток по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включить сетку на графике
        self._ax.grid()

        # Отобразить поле в начальный момент времени
        self._line, = self._ax.plot(self._xList, np.zeros(self.maxXSize))

    def drawProbes(self, probesPos: List[int]):
        '''
        Нарисовать пробники.

        probesPos - список координат пробников для регистрации временных
            сигналов (в отсчетах).
        '''
        # Отобразить положение пробника
        self._ax.plot(np.array(probesPos) * self._dx,
                      [0] * len(probesPos), self._probeStyle)

        for n, pos in enumerate(probesPos):
            self._ax.text(
                pos * self._dx,
                0,
                '\n{n}'.format(n=n + 1),
                verticalalignment='top',
                horizontalalignment='center')

    def drawSources(self, sourcesPos: List[int]):
        '''
        Нарисовать источники.

        sourcesPos - список координат источников (в отсчетах).
        '''
        # Отобразить положение пробника
        self._ax.plot(np.array(sourcesPos) * self._dx,
                      [0] * len(sourcesPos), self._sourceStyle)

    def drawBoundary(self, position: int):
        '''
        Нарисовать границу в области моделирования.

        position - координата X границы (в отсчетах).
        '''
        self._ax.plot([position * self._dx, position * self._dx],
                      [self.minYSize, self.maxYSize],
                      '--k')

    def stop(self):
        '''
        Остановить анимацию
        '''
        pylab.ioff()

    def updateData(self, data: npt.NDArray[float], timeCount: int):
        '''
        Обновить данные с распределением поля в пространстве
        '''
        self._line.set_ydata(data)
        time_str = '{:.5f}'.format(timeCount * self._dt * 1e9)
        self._ax.set_title(f'{time_str} нс')
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

###
### OBSOLETE
###        
def showProbeSignals(probes: List[Probe],
                     dx: float, dt: float, minYSize: float, maxYSize: float):
    '''
    Показать графики сигналов, зарегистрированых в датчиках.

    probes - список экземпляров класса Probe.
    dx - дискрет по пространству, м
    dt - дискрет по времени, с
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''
    # Создание окна с графиков
    fig, axes_list = pylab.subplots(nrows=len(probes),
                                    figsize=(10, 6.5),
                                    tight_layout={'w_pad': 0.7, 'h_pad': 0.7})
    fig.suptitle('Сигналы в датчиках')

    maxval = 0;
    minval = 0;

    # legends = []
    # Вывод сигналов в окно
    for n, (probe, ax) in enumerate(zip(probes, axes_list)):
        # Настройка внешнего вида графиков
        ax.set_xlim(0, len(probes[0].E) * dt * 1e9)
        ax.set_ylim(minYSize, maxYSize)
        ax.set_xlabel('t, нс')
        ax.set_ylabel('Ez, В/м')
        ax.grid()

        time_list = np.arange(len(probe.E)) * dt * 1e9
        maxval = np.max(probe.E)
        print(f'dx: {dx}')
        print(f"{n.numerator} max: {maxval}")
        minval = np.min(probe.E)
        legend = 'Датчик {n}: x = {pos:.5f}; Max = {maxval:.5f}; Min = {minval:.5f}'.format(
            n=n + 1, pos=probe.position * dx, maxval=maxval, minval=minval)
        # legends.append(legend)
        ax.plot(time_list, probe.E)

        # Создание и отображение легенды на графике
        legend_obj = ax.legend([legend])
        legend_obj.set_draggable(True)


    # Показать окно с графиками
    pylab.show()
  
    


def showProbeAll(probeSignal: Probe, probes: List[Probe],
                     dx: float, dt: float, minYSize: float, maxYSize: float):
    '''
    Показать графики сигналов, зарегистрированых в датчиках.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''

# Создание окна с графиков
    fig, axes_list = pylab.subplots(nrows=2 * len(probes) + 2,
                                    figsize=(10, 6.5),
                                    tight_layout={'w_pad': 0.7, 'h_pad': 0.7})
    fig.suptitle('Сигналы в датчиках')

    maxval = 0;
    minval = 0;

    b = len(probes)

    # Вывод сигналов в окно
    for n, (probe, ax) in enumerate(zip(probes, axes_list[range(0, b)])):
        # Настройка внешнего вида графиков
        ax.set_xlim(0, len(probes[0].E) * dt * 1e9)
        ax.set_ylim(minYSize, maxYSize)
        ax.set_xlabel('t, нс')
        ax.set_ylabel('Ez, В/м')
        ax.grid()

        time_list = np.arange(len(probe.E)) * dt * 1e9
        
        maxval = np.max(probe.E)
        minval = np.min(probe.E)
        
        legend = 'Датчик {n}: x = {pos:.5f} m; U Max = {maxval:.5f}; U Min = {minval:.5f}'.format(
            n=n + 1, pos=probe.position * dx, maxval=maxval, minval=minval)
        # legends.append(legend)
        ax.plot(time_list, probe.E)

        # Создание и отображение легенды на графике
        legend_obj = ax.legend([legend])
        legend_obj.set_draggable(True)

    
    

    # спектральная плотность
    for n, (probe, ax) in enumerate(zip(probes, axes_list[range(b, 2 * b)])):
        fft = np.abs(np.fft.fft(probe.E))
        #len(probe.E)
        fftfreq = np.fft.fftfreq(len(probe.E), dt)

        # Настройка внешнего вида графиков

        ax.set_xlim(np.min(fftfreq) / 150, np.max(fftfreq) / 150)
        ax.set_ylim(np.min(fft), np.max(fft))
        ax.set_xlabel('f, Hz')
        ax.set_ylabel('Ez, В/Гц')
        ax.grid()
        
        maxval = np.max(probe.E)

        minval = np.min(probe.E)
        legend = 'Датчик {n}: x = {pos:.5f} m; F Max = {maxval:.5f}; F Min = {minval:.5f}'.format(
            n=n + 1, pos=probe.position * dx, maxval=np.max(fft), minval=np.min(fft))
        # legends.append(legend)
        ax.plot(fftfreq, fft)

        # Создание и отображение легенды на графике
        legend_obj = ax.legend([legend])
        legend_obj.set_draggable(False)


    # ГРАФИК СПЕКТА ПАДАЮЩЕГО СИГНАЛА

    ax = axes_list[-2]
    # частота подающего сигнала


    #len(probe.E)
    fftfreq = np.fft.fftfreq(len(probes[1].E), dt)
    fft_P = np.abs(np.fft.fft(probeSignal.E))

    # Настройка внешнего вида графиков

    ax.set_xlim(np.min(fftfreq) / 300, np.max(fftfreq) / 300)
    ax.set_ylim(0, np.max(fft_P))
    ax.set_xlabel('f, Hz')
    ax.set_ylabel('Ez, В/Гц')
    ax.grid()
    
    ax.plot(fftfreq, fft_P)



    # ПОСЛЕДНИЙ ГРАФИК
    
    ax = axes_list[-1]
    # Гамма от частоты
    
    #частота отраженного сигнала
    fft_O = np.abs(np.fft.fft(probes[0].E))
    
    for i in range(len(fftfreq)):
        if fft_O[i] < 0.5:
            fft_O[i] = 0.0
        if fft_P[i] < 0.5:
            fft_P[i] = 0.0

    Gamma = np.abs( np.divide(
    fft_O,
    fft_P
    ) )


    # Настройка внешнего вида графиков

    ax.set_xlim(np.min(fftfreq) / 150, np.max(fftfreq) / 150)
    ax.set_ylim(0, 1)
    ax.set_xlabel('f, Hz')
    ax.set_ylabel('Ez, В/Гц')
    ax.grid()
    

    ax.plot(fftfreq, Gamma)

    pylab.show()
    
# Большие графики для отчёта
def showProbeFalRefSignal(probeFallSignal: Probe, probeRefSignal: Probe, dx: float, dt: float,
                      minYSize: float, maxYSize: float):
    '''
    Сформировать график по выбранному датчику.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''

# Создание окна с графиков
    fig, axes_list = pylab.subplots(nrows=2,
                                    figsize=(10, 6.5),
                                    tight_layout={'w_pad': 0.7, 'h_pad': 0.7})
    fig.suptitle('Сигналы')

    maxval = 0;
    minval = 0;
    
    probes = [probeFallSignal, probeRefSignal]

    # Вывод сигналов в окно
    for n, (probe, ax) in enumerate(zip(probes, axes_list[range(0, 2)])):
        # Настройка внешнего вида графиков
        ax.set_xlim(0, len(probes[n].E) * dt * 1e9)
        ax.set_ylim(minYSize, maxYSize)
        ax.set_xlabel('t, нс')
        ax.set_ylabel('Ez, В/м')
        ax.grid()

        time_list = np.arange(len(probe.E)) * dt * 1e9
        
        maxval = np.max(probe.E)
        minval = np.min(probe.E)
        
        legend = 'Датчик {n}: x = {pos:.5f} m; U Max = {maxval:.5f}; U Min = {minval:.5f}'.format(
            n=n + 1, pos=probe.position * dx, maxval=maxval, minval=minval)
        # legends.append(legend)
        ax.plot(time_list, probe.E)

        # Создание и отображение легенды на графике
        legend_obj = ax.legend([legend])
        legend_obj.set_draggable(True)

    pylab.show()
    
def showProbeFalRefSpectrum(probeFallSignal: Probe, probeRefSignal: Probe, dx: float, dt: float,
                      minYSize: float, maxYSize: float):
    '''
    Сформировать график по выбранному датчику.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''

# Создание окна с графиков
    fig, axes_list = pylab.subplots(nrows= 2,
                                    figsize=(10, 6.5),
                                    tight_layout={'w_pad': 0.7, 'h_pad': 0.7})
    fig.suptitle('Спектры сигналов')

    maxval = 0;
    minval = 0;
    
    probes = [probeFallSignal, probeRefSignal]

    for n, (probe, ax) in enumerate(zip(probes, axes_list[range(0, 2)])):
        fft = np.abs(np.fft.fft(probe.E))
        #len(probe.E)
        fftfreq = np.fft.fftfreq(len(probe.E), dt)

        # Настройка внешнего вида графиков

        ax.set_xlim(np.min(fftfreq) / 300, np.max(fftfreq) / 300)
        ax.set_ylim(np.min(fft), np.max(fft))
        ax.set_xlabel('f, Hz')
        ax.set_ylabel('Ez, В/Гц')
        ax.grid()
        
        maxval = np.max(probe.E)

        minval = np.min(probe.E)
        legend = 'Датчик {n}: x = {pos:.5f} m; F Max = {maxval:.5f}; F Min = {minval:.5f}'.format(
            n=n + 1, pos=probe.position * dx, maxval=np.max(fft), minval=np.min(fft))
        # legends.append(legend)
        ax.plot(fftfreq, fft)

        # Создание и отображение легенды на графике
        legend_obj = ax.legend([legend])
        legend_obj.set_draggable(False)

    pylab.show()
    
# Большие графики для отчёта
def showProbeKoeReflectedOfFrequency(probeFallSignal: Probe, probeRefSignal: Probe, dx: float, dt: float,
                      minYSize: float, maxYSize: float):
    '''
    Сформировать график по выбранному датчику.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''

# Создание окна с графиков
    fig, axes_list = pylab.subplots(nrows= 2,
                                    figsize=(10, 6.5),
                                    tight_layout={'w_pad': 0.7, 'h_pad': 0.7})
    fig.suptitle('Спектр отражённого сигнала')

    maxval = 0;
    minval = 0;

    # ГРАФИК СПЕКТРА ПАДАЮЩЕГО СИГНАЛА

    ax = axes_list[0]
    # частота падающего сигнала


    #len(probe.E)
    fftfreq = np.fft.fftfreq(len(probeFallSignal.E), dt)
    fft_P = np.abs(np.fft.fft(probeFallSignal.E))

    # Настройка внешнего вида графиков

    #ax.set_xlim(0, numpy.max(fftfreq)) -2*1e-9, 2*1e-9
    ax.set_xlim(np.min(fftfreq) / 300, np.max(fftfreq) / 300)
    #ax.set_xlim(-4*1e-9, 4*1e-9)
    ax.set_ylim(0, np.max(fft_P))
    ax.set_xlabel('f, Hz')
    ax.set_ylabel('Ez, В/Гц')
    ax.grid()
    
    ax.plot(fftfreq, fft_P)



    # ПОСЛЕДНИЙ ГРАФИК
    
    ax = axes_list[1]
    # Гамма от частоты
    
    #частота отраженного сигнала
    fft_O = np.abs(np.fft.fft(probeRefSignal.E))
    
    for i in range(len(fftfreq)):
        if fft_O[i] < 16:
            fft_O[i] = 0.0
        if fft_P[i] < 16:
            fft_P[i] = 0.0


    Gamma = np.abs( np.divide(
    fft_O,
    fft_P
    ) )


    # Настройка внешнего вида графиков

    #ax.set_xlim(0, numpy.max(fftfreq)) -2*1e-9, 2*1e-9
    ax.set_xlim(np.min(fftfreq) / 300, np.max(fftfreq) / 300)
    ax.set_ylim(0, 1)
    ax.set_xlabel('f, Hz')
    ax.set_ylabel('Ez, В/Гц')
    ax.grid()
    

    ax.plot(fftfreq, Gamma)

    pylab.show()


def showProbeFalRefSignalMy(probeFallSignal: Probe, probeRefSignal: Probe, dx: float, dt: float,
    minYSize: float, maxYSize: float):
    '''
    Сформировать график по выбранным датчикам.

    scss
    Copy code
    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''
    # Создание окна с графиками
    fig, ax = pylab.subplots(figsize=(10, 6.5))
    fig.suptitle('Сигналы')

    maxval = 0
    minval = 0

    probes = [probeFallSignal, probeRefSignal]

    # Вывод сигналов на одном графике
    for probe in probes:
        # Настройка внешнего вида графиков
        ax.set_xlim(0, len(probe.E) * dt * 1e9)
        ax.set_ylim(minYSize, maxYSize)
        ax.set_xlabel('t, нс')
        ax.set_ylabel('Ez, В/м')
        ax.grid()

        time_list = np.arange(len(probe.E)) * dt * 1e9

        maxval = np.max(probe.E)
        minval = np.min(probe.E)

        legend = 'Датчик: x = {pos:.5f} m; U Max = {maxval:.5f}; U Min = {minval:.5f}'.format(
            pos=probe.position * dx, maxval=maxval, minval=minval)
    
        linestyle = '--' if probe == probeRefSignal else '-'  # Changing line style for the second probe
        ax.plot(time_list, probe.E, linestyle=linestyle, label=legend)

    # Создание и отображение легенды на графике
    legend_obj = ax.legend()
    legend_obj.set_draggable(True)

    pylab.show()
    
def showProbeFalRefSpectrumMy(probeFallSignal: Probe, probeRefSignal: Probe, dx: float, dt: float,
                                minYSize: float, maxYSize: float):
    '''
    Сформировать график по выбранным датчикам.

    scss
    Copy code
    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    '''

    # Создание окна с графиками
    fig, ax = pylab.subplots(figsize=(10, 6.5), nrows=1, ncols=1)
    ax.set_title('Спектры сигналов')

    fftFall = np.abs(np.fft.fft(probeFallSignal.E))
    fftRef = np.abs(np.fft.fft(probeRefSignal.E))
    fftfreq = np.fft.fftfreq(len(probeFallSignal.E), dt)

    # Настройка внешнего вида графика
    ax.set_xlim(np.min(fftfreq) / 300, np.max(fftfreq) / 300)
    ax.set_ylim(min(min(fftFall), min(fftRef)), max(max(fftFall), max(fftRef)))
    ax.set_xlabel('f, Hz')
    ax.set_ylabel('Ez, В/Гц')
    ax.grid()

    maxFall = np.max(probeFallSignal.E)
    minFall = np.min(probeFallSignal.E)
    legendFall = 'Датчик 1: x = {pos:.5f} m; F Max = {maxval:.5f}; F Min = {minval:.5f}'.format(
        pos=probeFallSignal.position * dx, maxval=np.max(fftFall), minval=np.min(fftFall))
    ax.plot(fftfreq, fftFall, label=legendFall)

    maxRef = np.max(probeRefSignal.E)
    minRef = np.min(probeRefSignal.E)
    legendRef = 'Датчик 2: x = {pos:.5f} m; F Max = {maxval:.5f}; F Min = {minval:.5f}'.format(
        pos=probeRefSignal.position * dx, maxval=np.max(fftRef), minval=np.min(fftRef))
    ax.plot(fftfreq, fftRef, linestyle='--', label=legendRef)

    # Создание и отображение легенды на графике
    legend_obj = ax.legend(loc='best')
    legend_obj.set_draggable(False)

    pylab.show()