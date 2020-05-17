from . import coesa
from .base import EARTH_R, G_EARTH

       
def get_temperature(h):
    return coesa.table(h)[1]

def get_pressure(h):
    return coesa.table(h)[3]

def get_rho(h):
    return coesa.table(h)[3]

def get_mu(temp):
    a_0 = 1.47e-6
    S = 113
    if temp.any() > 1200:
        mu = a_0 * temp**(1.5)/(temp+S)  * (1+1.53e-4*(temp/S  -1)**2)
    else:
        mu = a_0 * temp**(1.5) / (temp + S)
    
    return mu


def get_TempPressRhoMu(h, method=1):
    temperature, pressure, rho = coesa.table(h)[1:]
    if method == 1:
        mu = get_mu(temperature)
    else:
        mu = 2.791e-7 * temperature ** (0.7355)
    return temperature, pressure, rho, mu

def get_g(h):
    return G_EARTH * (EARTH_R / (EARTH_R + h))**2
