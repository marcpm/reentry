t_span = [0.0, 2000];
initial_conds =  [23000*0.3048; 1.0*pi/180; 250000*0.3048; 0.0];

[t, y] = ode45(@lifting_odes, t_span, initial_conds);



%     for gamma in gamma_0s:
%         initial_conditions[1] = gamma # set gamma initial for ode
%         [t, y] = ode45(@lifting_odes, t_span=time_span, 
%                             y0=initial_conditions, args=[beta, get_rho, get_g, c_L, c_D], 
%                             dense_output=True, method=solver, atol=1e-6, rtol=1e-3)
% 
%         sol = result.sol(time_spans_dense)
%         _gamma = rad2deg(gamma)
% 
%         ax[0,0].plot(sol[2]/0.3048, sol[0]/0.3048, label=f"$\\gamma$={_gamma}" )
%         ax[0,0].set_xlabel("Altitude [ft]")
%         ax[0,0].set_ylabel("Velocity [ft/s]")
%         ax[0,1].plot(sol[2]/0.3048, time_spans_dense, label=f"$\\gamma$={_gamma}" )
%         ax[0,1].set_xlabel("Altitude [ft]")
%         ax[0,1].set_ylabel("Time [s]")
%         ax[1,0].plot(sol[2][:-1]/0.3048, np.diff(sol[0])/-9.81, label=f"$\\gamma$={_gamma}" )
%         ax[1,0].set_xlabel("Altitude [ft]")
%         ax[1,0].set_ylabel("Deceleration [g]")
% 






function  [ddt] = lifting_odes(t, x)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    V = x(1);
    gamma = x(2);
    h = x(3);
    r = x(4);
    c_L=0.84;
    c_D=0.84;
    beta = 4237.94150483602;
    EARTH_R = 6.3781e6; % m
    G_EARTH = 9.80665; % m/s^2
    [T, a, P, rho] = atmoscoesa(h);
    g = G_EARTH * (EARTH_R / (EARTH_R + h))^2;
    
    q = 0.5 * rho * V^2;
    
    dv_dt = g * (-q/beta  +  sin(gamma));
    dgamma_dt = ( -q*g / beta * c_L/c_D  + cos(gamma)*(g-V^2/(EARTH_R+h))) / (V);
    dh_dt = -1* V * sin(gamma);
    dr_dt = (EARTH_R * V * cos(gamma)) / (EARTH_R+h) ;


    ddt = [dv_dt; dgamma_dt; dh_dt; dr_dt];
end

