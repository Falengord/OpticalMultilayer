#IMPORT LIBRARIES
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

#roughness list in micron
z_1 = 0
z_2 = 0
z_4 = 0

#list of layer thicknesses in micron
d_GaN   = 5.57
d_Al2O3 = 500

#urbach energies, urbach zero energy and zero absorption coefficient
Eu_GaN = 0.017 #in eV
E0_GaN = 3.56  #in eV
alpha0_GaN = 5e3

Eu_LN = 0.150  #in eV
E0_LN = 4.502  #in eV
alpha0_LN = 3e3

#angle of incidence on the first layer
th = 0 * π/180

#load the measurement datafile to compare it with the simulation
x, y = np.loadtxt('Dati/7.1_hi3.txt', delimiter='\t', unpack=True)
    
xdata = x*1e-3  #in micron
ydata = y*1e-2


#execute the various operations for each wavelength
def func(λ,d_LN,z_3):
    
    #compute the refractive indeces of the various materials
    nreGaN = material_nk_fnGaN(λ, 'o') 
    nreLN  = material_nk_fnLN(λ, 'o')
    nimGaN = make_k(λ, alpha0_GaN, E0_GaN, Eu_GaN) 
    nimLN  = make_k(λ, alpha0_LN, E0_LN, Eu_LN)
    
    n_LN    = nreLN + 1j * nimLN   
    n_GaN   = nreGaN + 1j * nimGaN
    n_Al2O3 = material_nk_fnAl2O3(λ, 'o')
        
    #compute the transfer matrices for the various interfaces and layers, one by one
    T12 = make_T_int(λ, 1., n_LN, th, z_1)
    T2  = make_T_lay(λ, n_LN, d_LN, th)
    T23 = make_T_int(λ, n_LN, n_GaN, th, z_2)
    T3  = make_T_lay(λ, n_GaN, d_GaN, th)
    T34 = make_T_int(λ, n_GaN, n_Al2O3, th, z_3)
    T4  = make_T_lay(λ, n_Al2O3, d_Al2O3, th)
    T41 = make_T_int(λ, n_Al2O3, 1., th, z_4)
        
    temp = np.einsum('iak,ajk->ijk',T12,T2)
    temp = np.einsum('iak,ajk->ijk',temp,T23)
    temp = np.einsum('iak,ajk->ijk',temp,T3)
    temp = np.einsum('iak,ajk->ijk',temp,T34)
    temp = np.einsum('iak,ajk->ijk',temp,T4)

    return T_inc(xdata, temp, T41, n_Al2O3, d_Al2O3, th) 

popt, pcov = curve_fit(func, xdata, ydata, p0 = [0.23, 0.0151])
popt

fit = func(xdata, *popt)

plt.figure(1)
plt.plot(xdata, ydata, 'k-', alpha=0.5, linewidth=0.8, label='Experimental data')
plt.plot(xdata, fit, 'r-', label='fit: d_LN=%5.3f, z_3=%5.3f' % tuple(popt))
plt.xlabel('Wavelength (μm)')
plt.ylabel('Fraction of power transmitted')
plt.title('Transmission at normal incidence')
plt.legend()
plt.savefig('Transmission_7.1.pdf')
plt.show()

d = np.array([xdata,fit]).T
np.savetxt('simLN_7.1.txt', d, fmt='%10.5f', newline='\n')

chisquare(fit, ydata)

