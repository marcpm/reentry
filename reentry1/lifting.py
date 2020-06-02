import numpy as np
from scipy.integrate import solve_ivp
from math import sin, cos, pi, sqrt
# from numpy import sin, cos, pi
from . import coesa
import collections
from .base import EARTH_R, G_EARTH
from .base import (lbf2N, ft2m, m2ft, m2mi, deg2rad,
                    rad2deg, lbfsqf2Nsqm, Pa2lbfsqf, 
                    Btusqft2Wsqm, Wsqm2Btusqft)
from .atmos import (get_g, get_mu, get_pressure, get_rho,
                    get_temperature, get_TempPressRhoMu)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# plotting config
plt.style.use('classic')
font = {'family' : 'monospace',
        'weight' : 'regular',
        'size'   : 7}

plt.rc('font', **font) 
plt.rc('legend',fontsize=7)



def lifting_odes2(h, x, beta, rho, g, c_L, c_D):
    V = x[0]
    gamma = x[1]
    t = x[2]
    r = x[3]

    q = 0.5 * rho(h) * V**2
    
    dv_dh = g(h) * (-q/beta  +  sin(gamma)) * (-1 / ( V * sin(gamma)))
    dgamma_dh = (-1 / ( V * sin(gamma))) * ( -q*g(h) / beta * c_L/c_D  + cos(gamma)*(g(h)-V**2/(EARTH_R+h))) / (V)
    dt_dh = -1 / ( V * sin(gamma))
    dr_dh = (EARTH_R * V * cos(gamma)) / (EARTH_R+h) * (-1 / ( V * sin(gamma)))

    return [dv_dh, dgamma_dh, dt_dh, dr_dh]

def lifting_odes_marv(t, x, beta, rho, g, c_L, c_D):
    V = x[0]
    gamma = x[1]
    h = x[2]
    r = x[3]
    q = 0.5 * rho(h) * V**2
    if h >= ft2m(150_000):
        c_L = 0.0
    elif h < ft2m(250_000):
        c_L = 0.2
        c_D = 0.4
    dv_dt = g(h) * (-q/beta  +  sin(gamma))
    dgamma_dt = ( -q*g(h) / beta * c_L/c_D  + cos(gamma)*(g(h)-V**2/(EARTH_R+h))) / (V)
    dh_dt = -1* V * sin(gamma)
    dr_dt = (EARTH_R * V * cos(gamma)) / (EARTH_R+h) 

    return [dv_dt, dgamma_dt, dh_dt, dr_dt]

def lifting_odes(t, x, beta, rho, g, c_L, c_D):
    V = x[0]
    gamma = x[1]
    h = x[2]
    r = x[3]
    q = 0.5 * rho(h) * V**2
    
    dv_dt = g(h) * (-q/beta  +  sin(gamma))
    dgamma_dt = ( -q*g(h) / beta * c_L/c_D  + cos(gamma)*(g(h)-V**2/(EARTH_R+h))) / (V)
    dh_dt = -1* V * sin(gamma)
    dr_dt = (EARTH_R * V * cos(gamma)) / (EARTH_R+h) 

    return [dv_dt, dgamma_dt, dh_dt, dr_dt]
    
def run_lifting_simulation( beta=1291.72457,
                     V_0=23_000.0, gamma_0s=[0.1, 1.0, 2.5],  
                     altitude=250_000.0, time_elapsed = 2000.0, c_L=0.84, c_D=0.84,
                     spacecraft=False, input_units="imperial", plot_units="imperial", solver="RK45", marv=False):
    
    beta = lbfsqf2Nsqm(beta)
    if not isinstance(gamma_0s, collections.Iterable):
        gamma_0s = [gamma_0s]
    gamma_0s = map(deg2rad, gamma_0s)  # set gamma  deg2rad

    # convert altitude ft 2 m
    altitude = ft2m(altitude) if  input_units=="imperial" else altitude
    
    time_span = [0, time_elapsed] # seconds to integrate over

    V_0 = ft2m(V_0) if  input_units == "imperial" else V_0

    initial_conditions = [V_0, -1, altitude, 0.0] # initial conditions [V, gamma, altitude, range]
 
    time_spans_dense = np.linspace(*time_span, num=200) # sample time for later interpolation of solutions

    fig, axes = plt.subplots(4,3, figsize=(17,22),  ) # create plot figures

    for gamma in gamma_0s:
        initial_conditions[1] = gamma # set gamma initial for ode
        if not marv:
            result = solve_ivp(lifting_odes, t_span=time_span, 
                            y0=initial_conditions, args=[beta, get_rho, get_g, c_L, c_D], 
                            dense_output=True, method=solver, atol=1e-6, rtol=1e-3)
        else: 
            result = solve_ivp(lifting_odes_marv, t_span=time_span, 
                            y0=initial_conditions, args=[beta, get_rho, get_g, c_L, c_D], 
                            dense_output=True, method=solver, atol=1e-6, rtol=1e-3)
        sol = result.sol(time_spans_dense)
        # dump invalid solutions due unstiff solver sampling , integrating over negative altitudes
        time_spans_dense = time_spans_dense[ np.where(sol[2,:]>0)[0]]
        sol = sol[:, np.where(sol[2,:]>0)[0]]
        _gamma = rad2deg(gamma)

        # direct solutions
        v_sol =  sol[0]
        gamma_sol = sol[1]
        h_sol =  sol[2] # solution altitudes
        range_sol = sol[3]
        
        _altitudes  = m2ft(h_sol)/1e3 if plot_units == "imperial" else h_sol/1e3
        
        deccel_sol  = np.gradient(v_sol) / np.gradient(time_spans_dense) / -9.81 # in terms of g
        # atmospheric conditions
        temperatures, pressures, rhos, mus = get_TempPressRhoMu(h_sol) # array of atmospheric conditions at the dense altitude sampling

        L_ref = spacecraft.L_ref
        R_nose = spacecraft.R_nose

        mach_num = v_sol / np.sqrt(1.4 * 287 * temperatures)
        reynolds_num = rhos * v_sol * L_ref / mus
        dynamic_press = 0.5 * rhos * v_sol**2
        stagnation_heat = -1.83e-4 * np.sqrt(rhos/R_nose) * v_sol**3
        stagnation_pressure = rhos * v_sol**2 # assuming Cp=2 for a reentry capsule
        dynamic_energy = v_sol * dynamic_press
        K_i = 0.1235
        theta = 3055.5556
        gamma_heat = 1 + ((1.4-1) / (1+(1.4-1)*((theta/temperatures)**2 * (np.exp(theta/temperatures)/(np.exp(theta/temperatures)-1)**2))) )   # calorically imperfect gas gamma 
        c_p  = 1004.5  * (1 + (gamma_heat-1)/gamma_heat * (theta/temperatures)**2 * np.exp(theta/temperatures)/(np.exp(theta/temperatures)-1)**2)  
        h_wall = c_p * temperatures + v_sol**2 / 2
        # stagnation_enthalpy = -1*stagnation_heat*np.sqrt(R_nose/(stagnation_pressure/101325)/K_i)  + h_wall
        stagnation_enthalpy = c_p * temperatures + v_sol**2 / 2
       

        if plot_units == "imperial":
            dynamic_press = Pa2lbfsqf(dynamic_press)
            v_sol = m2ft(sol[0]) 
            gamma_sol = rad2deg(sol[1])
            time_sol = time_spans_dense
            range_sol = m2mi(sol[3])
            stagnation_heat = Wsqm2Btusqft(stagnation_heat)
            stagnation_pressure = Pa2lbfsqf(stagnation_pressure)
            stagnation_enthalpy *= 0.00043
            dynamic_energy = Wsqm2Btusqft(dynamic_energy)

        transparency = 0.4 if _gamma > 0.2 else 1.0
        axes[0,0].plot(_altitudes, v_sol, alpha=transparency, label=f"$\\gamma$={_gamma}" )
        axes[0,1].plot(_altitudes, deccel_sol, alpha=transparency, label=f"$\\gamma$={_gamma}" )
        axes[0,2].plot(_altitudes, dynamic_press, alpha=transparency, label=f"$\\gamma$={_gamma}")
    
        axes[1,0].plot(_altitudes, mach_num, alpha=transparency, label=f"$\\gamma$={_gamma}")
        axes[1,1].plot(_altitudes, reynolds_num, alpha=transparency, label=f"$\\gamma$={_gamma}")
        axes[1,2].plot(_altitudes, stagnation_pressure, alpha=transparency, label=f"$\\gamma$={_gamma}") ####
        
        axes[2,0].plot(_altitudes, stagnation_enthalpy, alpha=transparency, label=f"$\\gamma$={_gamma}")
        axes[2,1].plot(_altitudes, stagnation_heat, alpha=transparency, label=f"$\\gamma$={_gamma}")
        axes[2,2].plot(_altitudes, time_sol, alpha=transparency, label=f"$\\gamma$={_gamma}" )
        
        axes[3,0].plot(_altitudes, range_sol, alpha=transparency, label=f"$\\gamma$={_gamma}")
        axes[3,1].plot(_altitudes, dynamic_energy, alpha=transparency, label=f"$\\gamma$={_gamma}")
        axes[3,2].plot(_altitudes, gamma_heat)
    
    axes[0,0].set_ylabel("Velocity \n$[ft/s]$")
    axes[0,2].set_ylabel("Dynamic Pressure \n$[lbf / ft^{2}]$")
    axes[0,1].set_ylabel("Deceleration $[g]$")
    axes[1,0].set_ylabel("Mach Number")
    axes[1,1].set_ylabel("Reynolds Number")
    axes[1,2].set_ylabel("Stagnation Point \nPressure \n$[lbf/ ft^{2}]$")
    axes[2,0].set_ylabel("Stagnation Point \nEnthalpy \n$[Btu/lbm]$")
    axes[2,1].set_ylabel("Stagnation Point \nHeat Transfer \n$[Btu/s - ft^{2}]$")
    axes[2,2].set_ylabel("Entry Time $[s]$")
    axes[3,0].set_ylabel("Range $[mi]$")
    axes[3,1].set_ylabel("Dynamic Energy\n $[Btu/s-ft^{2}]$")

    for idx,ax in enumerate(axes.reshape(-1)):
        ax.set_xlabel("Altitude $[10^3 \;ft]$")
        ax.set_xticks(np.arange(0, 300, step=50))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(which="both")
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        # ax.legend()

    fig.delaxes(axes[3,2])
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.tight_layout()
    # plt.subplots_adjust( wspace=0.35, hspace=0.25, )
    plt.subplots_adjust( wspace= 1.05, hspace=0.55, )


    plt.show()



