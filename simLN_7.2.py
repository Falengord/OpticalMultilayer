#Paolo Wetzl 2020

#IMPORT LIBRERIE
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
    
from fun import material_nk_fnLN, material_nk_fnLNt, material_nk_fnGaN, material_nk_fnAl2O3, make_k, make_T_int, make_T_lay, T_inc

π = np.pi

'''
layer characteristics
1: air
2: LN
3: GaN
4: Al2O3
5: air 
''' 

#roughness list in microm
z_1 = 0
z_2 = 0
z_3 = 0.0151
z_4 = 0

# list of layer thicknesses in microm
d_LN = 0.330
d_GaN = 5.5754
d_Al2O3 = 500

#urbach energies, urbach zero energy and zero absorption coefficient
Eu_GaN = 0.017	#in eV
E0_GaN = 3.56	#in eV
alpha0_GaN = 16e3

Eu_LN = 0.200	#in eV
E0_LN = 4.502	#in eV
alpha0_LN = 4e2

#wavelenghts for the simulation in microm
l_sx = 0.350
l_dx = 2.5
n_points = 20000
lambda_list = np.linspace(l_sx, l_dx, n_points)

#Li composition in % 
cLi = 50

#initialization of T matrix
Tinc = []
	
#angle of incidence on the first layer
th = 0 * π/180

#execute the various operations for each wavelength
for λ in lambda_list:

    #compute the refractive indeces of the various materials
    nreGaN = material_nk_fnGaN(λ, 'o')
    nimGaN = make_k(λ, alpha0_GaN, E0_GaN, Eu_GaN)

    nreLN = material_nk_fnLN(λ,'o')
    nreLNt = material_nk_fnLNt(λ, 24.5, cLi,'o')
    nimLN = make_k(λ, alpha0_LN, E0_LN, Eu_LN)

    n_LN = nreLN + 1j * nimLN 
    n_LNt = nreLNt + 1j * nimLN    
    n_GaN = nreGaN + 1j * nimGaN
    n_Al2O3 = material_nk_fnAl2O3(λ, 'o')

    #compute the transfer matrices for the various interfaces and layers, one by one
    T12 = make_T_int(λ, 1., n_LN, th, z_1)
    T2  = make_T_lay(λ, n_LN, d_LN, th)
    T23 = make_T_int(λ, n_LN, n_GaN, th, z_2)
    T3  = make_T_lay(λ, n_GaN, d_GaN, th)
    T34 = make_T_int(λ, n_GaN, n_Al2O3, th, z_3)
    T4  = make_T_lay(λ, n_Al2O3, d_Al2O3, th)
    T41 = make_T_int(λ, n_Al2O3, 1., th, z_4)


    #use eq 16 (Katsidis) for the computation of the transmission coefficient of the multilayer from the total transfer 
    #matrix for the stack of layers
    Tinc.append(T_inc(λ, np.dot(np.dot(np.dot(np.dot(np.dot(T12,T2),T23),T3),T34),T4), T41, n_Al2O3, d_Al2O3, th))


#load the measurement datafile to compare it with the simulation
x, y = np.loadtxt('Dati/7.2 con bandwidth bassa.txt', delimiter='\t', unpack=True)



#save the result of the simulation on a file
d = np.array([lambda_list,Tinc]).T
np.savetxt('simLN_7.2.txt', d, fmt='%10.5f', newline='\n')


#plot the simulation and the measured data in the same graph
plt.figure(1)
plt.ylim(0, 1)
plt.plot(0.4135667662 * 2.99 / lambda_list, Tinc, 'g-', alpha=0.7, linewidth=0.8)
plt.plot(0.4135667662 * 2.99 / (x*1e-3), y*1e-2, 'b-', alpha=0.5, linewidth=0.8)
plt.xlabel('Energy (eV)')
plt.ylabel('Fraction of power transmitted')

plt.title('Transmission at normal incidence')
plt.savefig('Transmission_7.2.pdf')

#plot the data in order to be able to check the slope of the absorption edge (to estimate the Urbach energy)
plt.figure(2)
plt.xlim(3, 3.4)
plt.plot(0.4135667662 * 2.99 / lambda_list , np.log(-np.log(Tinc)), 'g-', alpha=0.7, linewidth=0.8)
plt.plot(0.4135667662 * 2.99 / (x*1e-3) , np.log(-np.log(y*1e-2)), 'b-', alpha=0.7, linewidth=0.8)
plt.xlabel('Energy (eV)')
plt.ylabel('ln(-ln(T))')

plt.show()
