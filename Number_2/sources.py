# -*- coding: utf-8 -*-
'''
Модуль с классами источников разного типа
'''

from abc import ABCMeta, abstractmethod

import numpy as np


class Source1D(metaclass=ABCMeta):
    '''
    Базовый класс для всех источников одномерного метода FDTD
    '''
    @abstractmethod
    def getE(self, time):
        '''
        Метод должен возвращать значение поля источника в момент времени time
        '''
        pass

    def getH(self, time):
        return 0.0


class SourcePlaneWave(metaclass=ABCMeta):
    @abstractmethod
    def getE(self, position, time):
        pass
    
    
class SourceTFSF(Source1D):
    def __init__(self, source: SourcePlaneWave,
                 sourcePos: float,
                 Sc: float = 1.0,
                 eps: float = 1.0,
                 mu: float = 1.0):
        self.source = source
        self.sourcePos = sourcePos
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        self.W0 = 120.0 * np.pi

    def getH(self, time):
        return -(self.Sc / (self.W0 * self.mu)) * self.source.getE(self.sourcePos, time)

    def getE(self, time):
        return (self.Sc / np.sqrt(self.eps * self.mu)) * self.source.getE(self.sourcePos - 0.5, time + 0.5)


class ModGauss(Source1D):
    def __init__(self, d, w, Nl, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d
        self.w = w
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu
        
    def getE(self, m, q):
        '''
        Расчет поля E в дискретной точке пространства m
        в дискретный момент времени q
        '''
        return (np.sin(2 * np.pi / self.Nl * (q * self.Sc - m * np.sqrt(self.eps * self.mu))) *
                np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w) ** 2))