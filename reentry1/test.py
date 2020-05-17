from reentry1.ballistic import ballistic_odes, run_ballistic_simulation
from reentry1.atmos import get_rho, get_g
from reentry1.base import  EARTH_R, lbfsqf2Nsqm, Pa2lbfsqf, Spacecraft
from reentry1.lifting import deg2rad , lbf2N, ft2m, run_lifting_simulation
from scipy.integrate import solve_ivp
from math import pi
import matplotlib.pyplot as plt
from reentry1.coesa import table

import numpy as np

def ballistic_example(mission="beta_study", plot_units="imperial"):
    if mission == "mercury_parachute":
        W = 2662.8  + 2* 80 + 3*103 # lbf
        L_ref = 6.2 #ft
        A_ref = 30.27 # ft^2 
        Cd = 1.60
        R_nose = 1.0 # ft
        imperial_initial_conditions = [23_000, 1.5, 0.0, 0.0] # [V_ini [ft], gamma_ini[deg], time_ini, range_ini]
        imperial_altitude = 250_000.0 # ft
        spacecraft = Spacecraft(W, A_ref, R_nose, L_ref, Cd, Cl=None, 
                                parachute=True, imperial_units=True, beta_study=False)   # spacecraft Class stores specs and handles unit conversions
        betas = Pa2lbfsqf(spacecraft.beta)
        print("Drogue Area", ft2m(ft2m(2* (23/2.2/2)**2*pi))) # ft^2 m^2)
        print("main area" , ft2m(ft2m(3* (116/2.2/2)**2 * pi))) 

    if mission == "mercury":
        W = 2662.8  # lbf
        L_ref = 6.2 #ft
        A_ref = 30.27 # ft^2 
        Cd = 1.60
        R_nose = 1.0 # ft
        imperial_initial_conditions = [23_000, 1.5, 0.0, 0.0] # [V_ini [ft], gamma_ini[deg], time_ini, range_ini]
        imperial_altitude = 250_000.0 # ft
        spacecraft = Spacecraft(W, A_ref, R_nose, L_ref, Cd, Cl=None, 
                                parachute=False, imperial_units=True, beta_study=False)   # spacecraft Class stores specs and handles unit conversions
        betas = Pa2lbfsqf(spacecraft.beta)

    elif mission == "beta_study":
        betas =  [100, 500, 1000, 5000]  # lbf/ft^2
        imperial_initial_conditions = [22_500, 12.0, 0.0, 0.0] # [V_ini [ft], gamma_ini[deg], time_ini, range_ini]
        imperial_altitude = 250_000.0 # ft
        spacecraft = Spacecraft(beta_study=True, L_ref=1.0, R_nose = 1.0, imperial_units=True)

    run_ballistic_simulation(betas=betas, 
                            initial_conditions=imperial_initial_conditions,
                            altitude=imperial_altitude, spacecraft=spacecraft,
                            input_units="imperial",
                                 plot_units=plot_units)

def lifting_example():

    W = 200_000 # lbf
    A_ref = 2690 # ft^2 m^2
    L_ref = 107.5 # ft
    R_nose = 1.0  # ft
    Cd = 0.84
    Cl = 0.84
    spacecraft = Spacecraft(W, A_ref, R_nose, L_ref, Cd, Cl, 
                                parachute=False, imperial_units=True, beta_study=False )

    beta = Pa2lbfsqf(spacecraft.beta)

    altitude = 250_000
    V_0= 23_000.0
    gamma_0s= [0.1, 1.0, 2.5]
    time_lapse = 1500 # 1300 LSODA
    run_lifting_simulation( beta=beta,
                     V_0=V_0, gamma_0s=gamma_0s,  
                     altitude=altitude, c_L=Cl, c_D=Cd, time_elapsed=time_lapse,
                     spacecraft=spacecraft,
                     input_units="imperial", solver="RK45")




def atmos_test():
    T =  lambda alt: 288.15 - 0.0065*alt
    p = lambda alt, T: 101325*(T/288.15)**5.256

    T_isa = T(ft2m(250_000))
    p_isa = p(ft2m(250_000), T_isa)

    rho_isa = p_isa/(287*T_isa)
    h = 269_000
    rho_76 = get_rho(ft2m(h))
    # print(f"@{ft2m(250000)} \nISA: \n   rho: {rho_isa}")
    print(f"@{h}ft \nUS 1976: \n   rho: {rho_76}")



if __name__ == "__main__":
    # ballistic_example("mercury_parachute")
    # ballistic_example("mercury")
    ballistic_example("beta_study", plot_units="metric")
    # lifting_example()



   