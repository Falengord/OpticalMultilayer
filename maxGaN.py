#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
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

#roughness list in micron
z_1 = 0.000#5
z_3 = 0

#list of layer thicknesses in micron
d_Al2O3 = 500

#urbach energies, urbach zero energy and zero absorption coefficient
Eu_GaN = 0.019 #in eV
alpha0_GaN = 5e3

#angle of incidence on the first layer
th = 0 * π/180

#load the measurement datafile to compare it with the simulation
x, y = np.loadtxt('Dati/n-GaN hi res detail 2.txt', delimiter='\t', unpack=True)

xdata = x*1e-3  #in micron
ydata = y*1e-2

#execute the various operations for each wavelength
def func1(λ, d_GaN, E0_GaN, z_2, a, b1, c1):
    
    A = a                     #3.6
    B = [b1,4.1]              #1.75, 4.1
    C = [c1,17.86**2]         #0.256**2, 17.86**2    
    
    #compute the refractive indeces of the various materials
    #nreGaN = material_nk_fnGaN(λ, 'o')
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

 #2*π*n/λ with variable b1 and c1 
def func2(λ,a,b1,c1,E0,z2):
    A = a
    B = [b1,4.1]
    C = [c1,17.86**2]
    return 2*π*n_sellmeier(λ,A,B,C)/λ

#Linear fit 
def linear(m,d_GaN,cost):
    return m*(π/(d_GaN))+cost

#Parabola
def parabola(x,l,j,k):
    return l*x**2 + j*x + k

#Linear fit 
def func3(m,d_GaN,cost,λ,a,b1,c1,E0,z2):
    return func2(λ,a,b1,c1,E0,z2)-linear(m,d_GaN,cost)

#Transmission interference fringe maxima

xdata = xdata[::-1] #reverse xdata (from 350nm to 850nm)
ydata = ydata[::-1] #reverse ydata

#initialization of various arrays
T_max = []          #array of maximum transmission value
lambda_max = []     #array of maximum lambda value
lambda_max_fit = [] #array of maximum lambda value (fit)
points_x = []       #array of xdata around a point
points_y = []       #array of ydata around a point
m = []              #array of integer numbers (m = 1,2,3...)
n = 1

i_list = list(range(100, len(xdata)-23))

for i in i_list:
    a = ydata[i]
    b1 = ydata[i+1]
    b2 = ydata[i+2]
    b3 = ydata[i+3]
    b4 = ydata[i+4]
    b5 = ydata[i+5]
    b6 = ydata[i+6]
    b7 = ydata[i+7]
    b8 = ydata[i+8]
    b9 = ydata[i+9]
    b10 = ydata[i+10]
    b11 = ydata[i+11]
    b12 = ydata[i+12]
    b13 = ydata[i+13]
    b23 = ydata[i+23]
    c1 = ydata[i-1]
    c2 = ydata[i-2]
    c3 = ydata[i-3]
    c4 = ydata[i-4]
    c5 = ydata[i-5]
    c6 = ydata[i-6]
    c7 = ydata[i-7]
    c8 = ydata[i-8]
    c9 = ydata[i-9]
    c10 = ydata[i-10]
    c11 = ydata[i-11]
    c12 = ydata[i-12]
    c13 = ydata[i-13]
                
    if (a>b1) & (a>b2) & (a>c1) & (a>c2):
            if i > 200:
                if (a>b3) & (a>b4) & (a>b5) & (a>c3) & (a>c4) & (a>c5):
                    if i > 2000:
                        if (a>b6) & (a>b7) & (a>b8) & (a>b9) & (a>c6) & (a>c7) & (a>c8) & (a>c9):
                            if i > 3500:
                                if (a>b10) & (a>b11) & (a>b12) & (a>b13) & (a>b23) & (a>c10) & (a>c11) & (a>c12) & (a>c13):
                                    
                                    for j in range(i-35,i+35):
                                        points_y.append(ydata[j])
                                        points_x.append(xdata[j])
                                    popt, pcov = curve_fit(parabola, points_x, points_y, p0=[1,1,1])
                                    max_fringe = -popt[1]/(2*popt[0])
                                    lambda_max_fit.append(max_fringe)
                                    points_x.clear()
                                    points_y.clear()
                                    
                                    T_max.append(ydata[i])
                                    lambda_max.append(xdata[i])
                                    m.append(n)
                                    n +=1
                            else:
                                
                                for j in range(i-15,i+15):
                                    points_y.append(ydata[j])
                                    points_x.append(xdata[j])
                                popt, pcov = curve_fit(parabola, points_x, points_y, p0=[1,1,1])
                                max_fringe = -popt[1]/(2*popt[0])
                                lambda_max_fit.append(max_fringe)
                                points_x.clear()
                                points_y.clear()
                                
                                T_max.append(ydata[i])
                                lambda_max.append(xdata[i])
                                m.append(n)
                                n +=1
                    else:
                        
                        for j in range(i-10,i+10):
                            points_y.append(ydata[j])
                            points_x.append(xdata[j])
                        popt, pcov = curve_fit(parabola, points_x, points_y, p0=[1,1,1])
                        max_fringe = -popt[1]/(2*popt[0])
                        lambda_max_fit.append(max_fringe)
                        points_x.clear()
                        points_y.clear()
                            
                        T_max.append(ydata[i])
                        lambda_max.append(xdata[i])
                        m.append(n)
                        n +=1
                        
            else:
                
                for j in range(i-3,i+3):
                    points_y.append(ydata[j])
                    points_x.append(xdata[j])
                popt, pcov = curve_fit(parabola, points_x, points_y, p0=[1,1,1])
                max_fringe = -popt[1]/(2*popt[0])
                lambda_max_fit.append(max_fringe)
                points_x.clear()
                points_y.clear()
                
                T_max.append(ydata[i])
                lambda_max.append(xdata[i])
                m.append(n)
                n +=1

                
m = m[::-1] #reverse m
# d = np.array([m,lambda_max,T_max]).T
# np.savetxt('maxGaN.txt', d, fmt='%10.6f', newline='\n')
d = np.array([m,lambda_max_fit]).T
np.savetxt('maxGaN.txt', d, fmt='%10.6f', newline='\n')

fig, ax = plt.subplots()
line, = plt.plot(lambda_max, T_max, '.', linewidth=1, label='Max (experimental data)')
line, = plt.plot(lambda_max_fit, T_max, '.', linewidth=1, label='Max (fit)')
line, = plt.plot(xdata, ydata, 'k-', alpha=0.7, linewidth=0.8, label='Experimental data')
ax.set_xlabel('Wavelength (μm)')
ax.set_ylabel('Fraction of power transmitted')
ax.set_title('Transmission at normal incidence')
plt.legend()
plt.show()

#ydata
q = func2(d[:,1],3.6,1.75,0.256**2,3.54,0.015)

#fit with linear
popt, pcov = curve_fit(linear, d[:,0], q, p0 = [5,15]) 
#plot of linear fit with optimized parameters
linear_fit = linear(d[:,0], *popt)

fig, ax = plt.subplots()
line, = plt.plot(d[:,0], q, '.', markersize=3, color='black', label='Max')
line2, = plt.plot(d[:,0], linear_fit, '-r', linewidth=1, label='d=%5.5f, cost=%5.5f' % tuple(popt))
plt.legend()
ax.set_xlabel('m')
ax.set_ylabel('q*n(λ_max) (1/μm)')
ax.set_xlim(0,55)
ax.set_ylim(15,55)

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)

#adjust the main plot to make room for the sliders
plt.subplots_adjust(top=0.975)
plt.subplots_adjust(bottom=0.33)
plt.subplots_adjust(left=0.1)
plt.subplots_adjust(right=0.95)

#make a horizontal slider to control a
axa = plt.axes([0.15, 0.04, 0.75, 0.015], facecolor=axcolor)
a_slider = Slider(
    ax=axa,
    label='a',
    valmin=2.0,
    valmax=4.5,
    valinit=3.6,
)

#make a horizontal slider to control b1
axb1 = plt.axes([0.15, 0.08, 0.75, 0.015], facecolor=axcolor)
b1_slider = Slider(
    ax=axb1,
    label='b1',
    valmin=0.5,
    valmax=5,
    valinit=1.75,
)

#make a horizontal slider to control c1
axc1 = plt.axes([0.15, 0.12, 0.75, 0.015], facecolor=axcolor)
c1_slider = Slider(
    ax=axc1,
    label='c1',
    valmin=0.0,
    valmax=0.35**2,
    valinit=0.256**2,
)

#make a horizontal slider to control E0
axE0 = plt.axes([0.15, 0.16, 0.75, 0.015], facecolor=axcolor)
E0_slider = Slider(
    ax=axE0,
    label='E0 [eV]',
    valmin=3.51,
    valmax=3.58,
    valinit=3.56,
    color="green",
)

#make a horizontal slider to control z2
axz2 = plt.axes([0.15, 0.2, 0.75, 0.015], facecolor=axcolor)
z2_slider = Slider(
    ax=axz2,
    label='z_2 [μm]',
    valmin=0.012,
    valmax=0.024,
    valinit=0.015,
    color="green",
)

#the function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(func2(d[:,1], a_slider.val, b1_slider.val, c1_slider.val, E0_slider.val, z2_slider.val))
    q = func2(d[:,1],a_slider.val,b1_slider.val,c1_slider.val,E0_slider.val, z2_slider.val) 
    #fit
    popt, pcov = curve_fit(linear, d[:,0], q, p0 = [5,15])
    pcov = np.sqrt(np.diag(pcov))
    
    #update fit
    line2.set_ydata(linear(d[:,0], *popt))
    #update legend()
    line2.set_label("d=({:.5}+/-{:.3}) μm, cost=({:.5}+/-{:.3}) 1/μm".format(float(popt[0]),float(pcov[0]),float(popt[1]),float(pcov[1])))
    ax.legend(loc='upper left')
    
    fig.canvas.draw_idle()
 
    fit = func1(xdata, popt[0], E0_slider.val, z2_slider.val, a_slider.val, b1_slider.val, c1_slider.val)
    
    fig2 = plt.figure(2)
    plt.clf()
    plt.plot(xdata, ydata, 'k-', alpha=0.7, linewidth=0.8, label='Experimental data')
    plt.plot(xdata, fit, 'r-', linewidth=0.8, label='Simulation')
    plt.legend()
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('T (%)')
    fig2.canvas.draw()
    plt.show()
    
    A = a_slider.val
    B = [b1_slider.val, 4.1]
    C = [c1_slider.val, 17.86**2]
    eqz_sell = n_sellmeier(xdata,A,B,C)
    ref = material_nk_fnGaN(xdata, 'o')
    
    fig3 = plt.figure(3)
    plt.clf()
    plt.plot(xdata, ref, 'k-', alpha=0.7, linewidth=0.8, label='Ref')
    plt.plot(xdata, eqz_sell, 'r-', alpha=0.7, linewidth=0.8, label='Simulation')
    plt.legend()
    plt.xlabel('Wavelength (μm)')
    plt.ylabel('n')
    plt.title('Sellmeier equation')
    fig3.canvas.draw()
    plt.show()
    
    res = func3(d[:,0], popt[0], popt[1], d[:,1], a_slider.val, b1_slider.val, c1_slider.val, E0_slider.val , z2_slider.val)
    
    fig4 = plt.figure(4)
    plt.clf()
    plt.plot(m, res, '.', color="black", markersize = 5,label = 'res')
    plt.legend()
    plt.xlabel('m')
    plt.ylabel('residui')
    fig4.canvas.draw()
    plt.show()
    
#register the update function with each slider
E0_slider.on_changed(update)
z2_slider.on_changed(update)
a_slider.on_changed(update)
b1_slider.on_changed(update)
c1_slider.on_changed(update)

plt.show()