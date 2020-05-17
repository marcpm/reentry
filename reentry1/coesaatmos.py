
from math import exp, sqrt
from skaero.atmosphere import coesa

EARTH_R = 6369.0 # m
GMR = 34.163195
NTAB = 8

HTAB  = [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852]
TTAB = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
PTAB = [1.0, 2.233611E-1, 5.403295E-2, 8.5666784E-3, 1.0945601E-3, 
                                     6.6063531E-4, 3.9046834E-5, 3.68501E-6]
GTAB = [-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0]


def atmos(altitude):
    h = altitude*EARTH_R/(altitude+EARTH_R)      #convert geometric to geopotential altitude
    i = 0
    j = NTAB                                       #setting up for binary search
    
    while True:
        k = (i+j) // 2                                              #integer division
        if h < HTAB[k]:
            j = k
        else:
            i = k
        if j <= i +1:
            break

    tgrad = GTAB[i]                                    # i will be in 1...NTAB-1
    tbase = TTAB[i]
    deltah = h - HTAB[i]
    tlocal = tbase + tgrad*deltah
    theta = tlocal / TTAB[1]                                    # temperature ratio

    if tgrad == 0.0:                                     # pressure ratio
        delta = PTAB[i] * exp(-GMR* deltah / tbase)
    else:
        delta = PTAB[i] * (tbase / tlocal) ** (GMR / tgrad)

    sigma = delta/theta                                           #! density ratio

    rho = sigma * 1.225
    pressure = delta * 101325
    temperature = theta * 288.15
    a = sqrt(1.4 * pressure/rho)
    # print()
    # print(f"speed of sound p/rho:  {a}")
    # b = sqrt(1.4 * 287 * (temperature))
    # print(f"speed of sound RT:  {b}")
    print()
    return altitude,  temperature, pressure, rho, a




print()
print(atmos(6.7056)) 
print()
print(coesa.table(0.6))