from math import pi
from . import coesa
# constants
EARTH_R = 6.3781e6 # m
G_EARTH = 9.80665 # m/s^2
GMR = 34.163195
NTAB = 8

class Spacecraft:

    def __init__(self,  W=0.0, A_ref=0.0, R_nose=0.0, L_ref=0.0, Cd=0.8, Cl=0, 
                parachute=False, imperial_units=True, beta_study=False,):
        
        self.beta_study = beta_study
        self.A_ref = A_ref
        self.Cd = Cd
        self.Cl = Cl
        self.R_nose = R_nose
        self.W = W
        self.L_ref = L_ref
        self.parachute = parachute
        if imperial_units :
            self.R_nose = ft2m(self.R_nose)
            self.W  = lbf2N(self.W)
            self.A_ref = ft2m(ft2m(self.A_ref))
            self.L_ref = ft2m(self.L_ref)
        self.beta = self.compute_beta()

    def compute_beta(self):
        if self.beta_study:
            return None
        return (self.W) / (self.Cd * self.A_ref)

class Scenario:
    def __init__(self, h_0, v_0, gamma_0):
        self.h_0 = h_0
        self.v_0 = v_0
        self.gamma_0 = gamma_0

# SI - Imperial Unit Conversion funcs
def lbf2N(x):
    return x * 4.448222

def ft2m(x):
    return x * 0.3048

def m2ft(x):
    return x / 0.3048

def m2mi(x):
    return x / 1.609344e3

def deg2rad(x):
    return x * pi / 180 

def rad2deg(x):
    return x * 180/pi

def lbfsqf2Nsqm(x):
#Beta from imperial to SI 
    return x * 47.880259

def Pa2lbfsqf(x):
    """Beta from SI to imperial"""
    return x / 47.880259

def Btusqft2Wsqm(x):
    return x * 11356.538527

def Wsqm2Btusqft(x):
    return x / 11356.538527
