import numpy as np # used for vectorized sin, cos, funcs
from scipy.integrate import solve_ivp # ode solver
from math import sin, cos, pi, sqrt # basic math imports

# local imports 
from . import coesa
from .base import EARTH_R, G_EARTH
from .base import (lbf2N, ft2m, m2ft, m2mi, deg2rad,
                    rad2deg, lbfsqf2Nsqm, Pa2lbfsqf, 
                    Btusqft2Wsqm, Wsqm2Btusqft)
from .atmos import (get_g, get_mu, get_pressure, get_rho,
                    get_temperature, get_TempPressRhoMu)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import collections

# plotting config
plt.style.use('classic')
font = {'family' : 'monospace',
        'weight' : 'regular',
        'size'   : 8}

plt.rc('font', **font) 
plt.rc('legend',fontsize=7)

def parachute_beta_mod(h, spacecraft=None ):
    # Default values for Mercury Capsule 1960s with Orion2020 Parachute System

    if ft2m(9000) < h < ft2m(25_000):
    # deploy 2 drogue parachutes
        W = spacecraft.W  # lbf
        A_ref = ft2m(ft2m(2* (23/2.2/2)**2*pi)) # ft^2 m^2
        c_D = 1.75
        beta = (W)/(c_D * A_ref)
        return beta
    if h <= ft2m(9000):
        # deploy 3 main parachutes
        W = spacecraft.W  # lbf
        A_ref = ft2m(ft2m(3* (116/2.2/2)**2 * pi)) # ft^2 m^2
        c_D = 1.85
        beta = (W)/(c_D * A_ref) 
        return beta
    return spacecraft.beta

def ballistic_odes(h, x, beta, rho, g, spacecraft=False):
    V = x[0]
    gamma = x[1]
    t = x[2]
    r = x[3]

    if spacecraft.parachute:
        beta = parachute_beta_mod(h, spacecraft)

    q = 0.5 * rho(h) * V**2

    dv_dh = (g(h) *(q/beta - sin(gamma))) / (V * sin(gamma))
    dgamma_dh = (cos(gamma) * (V**2/(EARTH_R+h) - g(h))) / (V**2 * sin(gamma))
    dt_dh = -1 / (V * sin(gamma))
    dr_dh = (-EARTH_R * cos(gamma)) / ((EARTH_R+h) * sin(gamma))

    return [dv_dh, dgamma_dh, dt_dh, dr_dh]

def run_ballistic_simulation( betas=[100,500,1000,5000],spacecraft=None,
                     initial_conditions=[22_500, 12, 0.0, 0.0], 
                     altitude=250_000,
                     input_units="imperial", solver="RK45", plot_units="imperial"):

    if not isinstance(betas, collections.Iterable):
        betas = [betas]

    altitude_range = [ft2m(altitude), 0.0] if input_units=="imperial" else [altitude, 0.0]

    if input_units == "imperial":
        betas = map(lbfsqf2Nsqm, betas) #    lbf/ft^2 to N/m^2
        initial_conditions[0] = ft2m(initial_conditions[0]) # convert initial velocity from ft/s to m/s

    initial_conditions[1] = deg2rad(initial_conditions[1])  # set gamma  deg2rad
    
    altitudes_dense = np.linspace(*altitude_range, num=200) # sample altitude values for plotting
    fig, axes = plt.subplots(4,3, figsize=(17,22),  ) # create plot figures

    for beta in betas:
        result = solve_ivp(ballistic_odes, t_span=altitude_range, 
                            y0=initial_conditions, args=[beta, get_rho, get_g, spacecraft], 
                            dense_output=True, method=solver)

        sol = result.sol(altitudes_dense)
        _altitudes  = m2ft(altitudes_dense)/1e3 if plot_units == "imperial" else altitudes_dense/1e3
        _beta = int(Pa2lbfsqf(beta)) if plot_units == "imperial" else int(beta)

        # direct solutions
        v_sol =  sol[0]
        gamma_sol = sol[1]
        time_sol =  sol[2]
        range_sol = sol[3]
        print(f"\nRun for beta: {_beta}")
        print(f" Touchdown velocity: {v_sol[-1]:.2f} m/s")
        print(f" Range: {range_sol[-1]:.2f} m")
        print(f" Entry Time: {time_sol[-1]:.2f} s")

        deccel_sol  = np.gradient(v_sol)/ np.gradient(time_sol) / -9.81 # in terms of g

        # atmospheric conditions
        temperatures, pressures, rhos, mus = get_TempPressRhoMu(altitudes_dense) # array of atmospheric conditions at the dense altitude sampling

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
            time_sol = sol[2]
            range_sol = m2mi(sol[3])
            stagnation_heat = Wsqm2Btusqft(stagnation_heat)
            stagnation_pressure = Pa2lbfsqf(stagnation_pressure)
            stagnation_enthalpy *= 0.00043
            dynamic_energy = Wsqm2Btusqft(dynamic_energy)

            axes[0,0].plot(_altitudes, v_sol, label=f"$\\beta$={_beta}" )
            axes[0,1].plot(_altitudes, deccel_sol, label=f"$\\beta$={_beta}" )
            axes[0,2].plot(_altitudes, dynamic_press, label=f"$\\beta$={_beta}")
        
            axes[1,0].plot(_altitudes, mach_num, label=f"$\\beta$={_beta}")
            axes[1,1].plot(_altitudes, reynolds_num, label=f"$\\beta$={_beta}")
            axes[1,2].plot(_altitudes, stagnation_pressure, label=f"$\\beta$={_beta}") ####
            
            axes[2,0].plot(_altitudes, stagnation_enthalpy, label=f"$\\beta$={_beta}")
            axes[2,1].plot(_altitudes, stagnation_heat, label=f"$\\beta$={_beta}")
            axes[2,2].plot(_altitudes, time_sol, label=f"$\\beta$={_beta}" )
            
            axes[3,0].plot(_altitudes, range_sol, label=f"$\\beta$={_beta}")
            axes[3,1].plot(_altitudes, dynamic_energy, label=f"$\\beta$={_beta}")
            axes[3,2].plot(_altitudes, gamma_heat)
        else:
            axes[0,0].plot(v_sol,_altitudes, label=f"$\\beta$={_beta}" )
            axes[0,1].plot(deccel_sol,_altitudes, label=f"$\\beta$={_beta}" )
            axes[0,2].plot(dynamic_press,_altitudes, label=f"$\\beta$={_beta}")
        
            axes[1,0].plot(mach_num,_altitudes, label=f"$\\beta$={_beta}")
            axes[1,1].plot(reynolds_num,_altitudes, label=f"$\\beta$={_beta}")
            axes[1,2].plot(stagnation_pressure,_altitudes, label=f"$\\beta$={_beta}") ####
            
            axes[2,0].plot(stagnation_enthalpy,_altitudes, label=f"$\\beta$={_beta}")
            axes[2,1].plot(stagnation_heat,_altitudes, label=f"$\\beta$={_beta}")
            axes[2,2].plot(time_sol,_altitudes, label=f"$\\beta$={_beta}" )
            
            axes[3,0].plot(range_sol,_altitudes, label=f"$\\beta$={_beta}")
            axes[3,1].plot(dynamic_energy,_altitudes, label=f"$\\beta$={_beta}")
            axes[3,2].plot(gamma_heat,_altitudes)    

    if plot_units == "imperial":
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
    else:
        axes[0,0].set_xlabel("Velocity \n$[m/s]$")
        axes[0,2].set_xlabel("Dynamic Pressure \n$[Pa]$")
        axes[0,1].set_xlabel("Deceleration $[g]$")
        axes[1,0].set_xlabel("Mach Number")
        axes[1,1].set_xlabel("Reynolds Number")
        axes[1,2].set_xlabel("Stagnation Point \nPressure \n$[Pa]$")
        axes[2,0].set_xlabel("Stagnation Point \nEnthalpy \n$[J/kg]$")
        axes[2,1].set_xlabel("Stagnation Point \nHeat Transfer \n$[W/m^{2}]$")
        axes[2,2].set_xlabel("Entry Time $[s]$")
        axes[3,0].set_xlabel("Range $[m]$")
        axes[3,1].set_xlabel("Dynamic Energy\n $[W/m^{2}]$")




    for idx,ax in enumerate(axes.reshape(-1)):
        if plot_units == "imperial":
            ax.set_xlabel("Altitude $[10^3 \;ft]$")
            ax.set_xticks(np.arange(0, 300, step=50))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))


        
        else:
            ax.set_ylabel("Altitude $[km]$")
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))

        
        ax.grid(which="both")
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0);
        # ax.legend()

    fig.delaxes(axes[3,2])

    # fig.tight_layout()
    plt.subplots_adjust( wspace= 1.05, hspace=0.55, )

    plt.show()
