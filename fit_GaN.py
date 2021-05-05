#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import chisquare

from fun import *
%matplotlib qt

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
#z_2 = 0.015
z_3 = 0

# list of layer thicknesses in micron
d_GaN = 5.57
d_Al2O3 = 430

#urbach energies, urbach zero energy and zero absorption coefficient
Eu_GaN = 0.017 #in eV
#E0_GaN = 3.56  #in eV
alpha0_GaN = 5e3

#angle of incidence on the first layer
th = 0 * π/180

#load the measurement datafile to compare it with the simulation
x, y = np.loadtxt('Dati/n-GaN hi res detail.txt', delimiter='\t', unpack=True)

xdata = x*1e-3  #in micron
ydata = y*1e-2

#execute the various operations for each wavelength
def func(λ,d_GaN, E0_GaN, z_2):  
    
    #compute the refractive indeces of the various materials
    nreGaN = material_nk_fnGaN(λ, 'o')
    nimGaN = make_k(λ, alpha0_GaN, E0_GaN, Eu_GaN) 
     
    n_GaN   = nreGaN + 1j * nimGaN
    n_Al2O3 = material_nk_fnAl2O3(λ, 'o')
        
    #compute the transfer matrices for the various interfaces and layers, one by one
    T12 = make_T_int(λ, 1., n_GaN, th, z_1)
    T2  = make_T_lay(λ, n_GaN, d_GaN, th)
    T23 = make_T_int(λ, n_GaN, n_Al2O3, th, z_2)
    T3  = make_T_lay(λ, n_Al2O3, d_Al2O3, th)
    T31 = make_T_int(λ, n_Al2O3, 1., th, z_3)
        
    temp = np.einsum('iak,ajk->ijk',T12,T2)
    temp = np.einsum('iak,ajk->ijk',temp,T23)
    temp = np.einsum('iak,ajk->ijk',temp,T3)

    return T_inc(xdata, temp, T31, n_Al2O3, d_Al2O3, th) 


#### Set seaborn ####
sns.set_theme()
sns.set_style("darkgrid")

#fit
popt, pcov = curve_fit(func, xdata, ydata, p0 = [5.56,3.56,0.015], maxfev=10000)                  

print(popt)
print(np.sqrt(np.diag(pcov)))

fit = func(xdata, *popt)

plt.figure(1)
plt.plot(xdata, fit, 'r-', label='Fit')
plt.plot(xdata, ydata, 'k-', alpha=0.5, linewidth=0.8, label='Experimental data')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Fraction of power transmitted')
plt.title('Transmission at normal incidence')
plt.legend()
plt.savefig('Transmission_GaN.pdf')
plt.show()

d = np.array([xdata,fit]).T
np.savetxt('simLN_GaN.txt', d, fmt='%10.5f', newline='\n')

print(chisquare(fit, ydata))