#Paolo Wetzl 2020

#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
    
from fun import material_nk_fnLN, material_nk_fnLNt, material_nk_fnGaN, material_nk_fnAl2O3, make_k, make_T_int, make_T_lay, absorption, T_inc

π = np.pi

'''
layer characteristics
1: air
2: GaN
3: Al2O3
4: air 
''' 

#roughness list in micron
z_1 = 0.000#5
z_2 = 0.015
z_3 = 0


# list of layer thicknesses in micron
d_GaN = 5.57
d_Al2O3 = 500


#urbach energies, urbach zero energy and zero absorption coefficient
Eu_GaN = 0.019 #in eV
E0_GaN = 3.56  #in eV from "Urbach–Martienssen tails in a wurtzite GaN epilayer", Chichibu 1997
alpha0_GaN = 5e3


#wavelenghts for the simulation in micron
l_sx = 0.350
l_dx = 2.5
n_points = 20000
lambda_list = np.linspace(l_sx, l_dx, n_points)


#initialization of various things matrices
T = []
Tinc = []
a_GaN = []

#angle of incidence on the first layer
th = 0 * π/180


#execute the various operations for each wavelength
for λ in lambda_list:

    #compute the refractive indeces of the various materials
    nreGaN = material_nk_fnGaN(λ,'o')
    nimGaN = make_k(λ, alpha0_GaN, E0_GaN, Eu_GaN)
    a_GaN.append(absorption(λ, alpha0_GaN, E0_GaN, Eu_GaN))
    
    n_GaN = nreGaN + 1j * nimGaN
    n_Al2O3 = material_nk_fnAl2O3(λ, 'o')

    #compute the transfer matrices for the various interfaces and layers, one by one
    T12 = make_T_int(λ, 1, n_GaN, th, z_1)
    T2  = make_T_lay(λ, n_GaN, d_GaN, th)
    T23 = make_T_int(λ, n_GaN, n_Al2O3, th, z_2)
    T3  = make_T_lay(λ, n_Al2O3, d_Al2O3, th)
    T31 = make_T_int(λ, n_Al2O3, 1, th, z_3)

    #use eq 16 (Katsidis) for the computation of the transmission coefficient of the multilayer from the total transfer matrix for 
    #the stack of layers
    Tinc.append(T_inc(λ, np.dot(np.dot(T12,T2),T23), T31, n_Al2O3, d_Al2O3, th))


#load the measurement datafile to compare it with the simulation
x, y = np.loadtxt('Dati/n-GaN hi res detail.txt', delimiter='\t', unpack=True)
x2, y2 = np.loadtxt('Dati/n-GaN.txt', delimiter='\t', unpack=True)

d = np.array([lambda_list,Tinc]).T
np.savetxt('simGaN.txt', d, fmt='%10.5f', newline='\n')

e = np.array([0.4135667662 * 2.99 / (x*1e-3), np.log(-np.log(y*1e-2))]).T
np.savetxt('simGaN_u.txt', e, fmt='%10.5f', newline='\n')


#plot the simulation and the measured data in the same graph
plt.figure(1)
plt.plot(0.4135667662 * 2.99 / lambda_list, Tinc, 'g-', alpha=0.7, linewidth=0.8)
plt.plot(0.4135667662 * 2.99 / (x2*1e-3), y2*1e-2, 'b-', alpha=0.5, linewidth=0.8)
plt.xlabel('Energy (eV)')
plt.ylabel('Fraction of power transmitted')

plt.title('Transmission at normal incidence')
plt.savefig('Transmission_GaN.svg')

plt.figure(2)
plt.xlim(3, 3.4)
plt.ylim(-2, 0)
plt.plot(0.4135667662 * 2.99 / lambda_list , np.log(-np.log(Tinc)), 'g-', alpha=0.7, linewidth=0.8)
plt.plot(0.4135667662 * 2.99 / (x2*1e-3) , np.log(-np.log(y2*1e-2)), 'b-', alpha=0.7, linewidth=0.8)
plt.xlabel('Energy (eV)')
plt.ylabel('ln(-ln(T))')

plt.show()