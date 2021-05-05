#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from matplotlib.widgets import Slider, Button

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

#roughness between Al2O3 and air in micron
z_3 = 0

#layer thicknesses in micron
d_Al2O3 = 500

#urbach energies, urbach zero energy and zero absorption coefficient
alpha0_GaN = 5e3

#angle of incidence on the first layer
th = 0 * π/180

#load the measurement datafile to compare it with the simulation
x, y = np.loadtxt('Dati/n-GaN hi res detail.txt', delimiter='\t', unpack=True)

xdata = x*1e-3  #in micron
ydata = y*1e-2

#execute the various operations for each wavelength
def func(λ,d_GaN, E0_GaN, Eu_GaN, z_1, z_2, a, b1, c1):
    
    A = a                     #3.6
    B = [b1,4.1]              #1.75, 4.1
    C = [c1,17.86**2]         #0.256**2, 17.86**2    
    
    #compute the refractive indeces of the various materials
    nreGaN = n_sellmeier(λ,A,B,C)
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

fit = func(xdata,5.56,3.56,0.019,0.0,0.015,3.6,1.75,0.256**2)

fig, ax = plt.subplots()
plt.plot(xdata, ydata, 'k-', alpha=0.7, linewidth=0.8, label='Experimental data')
line, = plt.plot(xdata, fit, 'r-', linewidth=0.8, label='Simulation')
plt.legend()
ax.set_xlabel('Wavelength (μm)')
ax.set_ylabel('Fraction of power transmitted')

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(top=0.975)
plt.subplots_adjust(bottom=0.46)

# Make a horizontal slider to control the thickness of GaN.
axd = plt.axes([0.2, 0.1, 0.7, 0.015], facecolor=axcolor)
d_slider = Slider(
    ax=axd,
    label='d_GaN [μm]',
    valmin=4,
    valmax=10,
    valinit=5.57,
    color="orange",
)

# Make a horizontal slider to control E0_GaN
axE0 = plt.axes([0.2, 0.135, 0.7, 0.015], facecolor=axcolor)
E0_slider = Slider(
    ax=axE0,
    label='E0_GaN [eV]',
    valmin=3.51,
    valmax=3.58,
    valinit=3.56,
    color="green",
)

# Make a horizontal slider to control Eu_GaN
axEu = plt.axes([0.2, 0.17, 0.7, 0.015], facecolor=axcolor)
Eu_slider = Slider(
    ax=axEu,
    label='Eu_GaN [eV]',
    valmin=0.012,
    valmax=0.022,
    valinit=0.019,
    color="green",
)

# Make a horizontal slider to control z_1
axz1 = plt.axes([0.2, 0.205, 0.7, 0.015], facecolor=axcolor)
z1_slider = Slider(
    ax=axz1,
    label='z_1 [μm]',
    valmin=0,
    valmax=0.02,
    valinit=0,
)

# Make a horizontal slider to control z_2
axz2 = plt.axes([0.2, 0.24, 0.7, 0.015], facecolor=axcolor)
z2_slider = Slider(
    ax=axz2,
    label='z_2 [μm]',
    valmin=0.012,
    valmax=0.024,
    valinit=0.015,
)

# Make a horizontal slider to control a
axa = plt.axes([0.2, 0.275, 0.7, 0.015], facecolor=axcolor)
a_slider = Slider(
    ax=axa,
    label='a',
    valmin=3,
    valmax=4.2,
    valinit=3.6,
    color='yellow',
)

# Make a horizontal slider to control b1
axb1 = plt.axes([0.2, 0.31, 0.7, 0.015], facecolor=axcolor)
b1_slider = Slider(
    ax=axb1,
    label='b1',
    valmin=1,
    valmax=2.2,
    valinit=1.75,
    color='yellow',
)

# Make a horizontal slider to control c1
axc1 = plt.axes([0.2, 0.345, 0.7, 0.015], facecolor=axcolor)
c1_slider = Slider(
    ax=axc1,
    label='c1',
    valmin=0.2**2,
    valmax=0.3**2,
    valinit=0.256**2,
    color='yellow',
)

# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(func(xdata, d_slider.val, E0_slider.val, Eu_slider.val, z1_slider.val, z2_slider.val, a_slider.val, b1_slider.val, c1_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
d_slider.on_changed(update)
E0_slider.on_changed(update)
Eu_slider.on_changed(update)
z1_slider.on_changed(update)
z2_slider.on_changed(update)
a_slider.on_changed(update)
b1_slider.on_changed(update)
c1_slider.on_changed(update)

plt.show()