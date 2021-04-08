#Paolo Wetzl 2020

#IMPORT LIBRARIES
from numpy import linspace, array, loadtxt, exp, dot, log, log10, savetxt, transpose
import matplotlib.pyplot as plt
    
from fun import material_nk_fnGaN, material_nk_fnAl2O3, make_k, make_T_int, make_T_lay, make_layer_matr, T_inc, absorption

#layer characteristics

#1: air
#2: GaN
#3: Al2O3
#4: air 


#valori buoni al 17/6/2020

#roughness list in microm
z_1 = 0.000#5
z_2 = 0.015
z_3 = 0


# list of layer thicknesses in microm
d_GaN = 5.57
d_Al2O3 = 500


#urbach energies, urbach zero energy and zero absorption coefficient
Eu_GaN = 0.019 #in eV
E0_GaN = 3.56  #in eV from "Urbach–Martienssen tails in a wurtzite GaN epilayer", Chichibu 1997
alpha0_GaN = 5e3


#wavelenghts for the simulation in microm
l_sx = 0.350
l_dx = 2.5
n_points = 20000
lambda_list = linspace(l_sx, l_dx, n_points)


#initialization of various things matrices
T = []
Tinc = []
a_GaN = []
	
#angles of incidence on the layers	
th_i = 0
th_f = 0


#execute the various operations for each wavelength
for lambda_vac in lambda_list:


	#compute the refractive indeces of the various materials
	nreGaN = material_nk_fnGaN(lambda_vac,'o')
	nimGaN = make_k(lambda_vac, alpha0_GaN, E0_GaN, Eu_GaN)
	a_GaN.append( absorption(lambda_vac, alpha0_GaN, E0_GaN, Eu_GaN) )
	    
	n_GaN = nreGaN + 1j * nimGaN
	n_Al2O3 = material_nk_fnAl2O3(lambda_vac, 'o')

	#compute the transfer matrices for the various interfaces and layers, one by one

	T12 = make_T_int(lambda_vac, 1, n_GaN, th_i, th_f, z_1)
	T2  = make_T_lay(lambda_vac, n_GaN, d_GaN)	
	T23 = make_T_int(lambda_vac, n_GaN, n_Al2O3, th_i, th_f, z_2)
	T3  = make_T_lay(lambda_vac, n_Al2O3, d_Al2O3)			
	T31 = make_T_int(lambda_vac, n_Al2O3, 1, th_i, th_f, z_3)
	
	#use eq 16 (Katsidis) for the computation of the transmission coefficient of the multilayer from the total transfer matrix for the stack of layers
	Tinc.append(T_inc(lambda_vac, dot(dot(T12,T2),T23), T31, n_Al2O3, d_Al2O3))



#load the measurement datafile to compare it with the simulation
x, y = loadtxt('Dati/n-GaN hi res detail.txt', delimiter='\t', unpack=True)
x2, y2 = loadtxt('Dati/n-GaN.txt', delimiter='\t', unpack=True)

d =array([lambda_list,Tinc]).T
#print(d)
savetxt('simGaN.txt', d, fmt='%10.5f', newline='\n')

e =array([0.4135667662 * 2.99 / (x/1000),log(-log(y / 100))]).T
savetxt('simGaN_u.txt', e, fmt='%10.5f', newline='\n')



#plot the simulation and the measured data in the same graph
plt.figure(1)
plt.plot(0.4135667662 * 2.99 / lambda_list, Tinc, 'g-', alpha=0.7, linewidth=0.8)
plt.plot(0.4135667662 * 2.99 / (x2/1000), y2 / 100, 'b-', alpha=0.5, linewidth=0.8)
plt.xlabel('Energy (eV)')
plt.ylabel('Fraction of power transmitted')

plt.title('Transmission at normal incidence')
plt.savefig('Transmission_GaN.svg')
plt.savefig('Transmission_GaN.pdf')

plt.figure(2)
plt.xlim(3, 3.4)
plt.ylim(-2, 0)
plt.plot(0.4135667662 * 2.99 / lambda_list , log(-log(Tinc)), 'g-', alpha=0.7, linewidth=0.8)
plt.plot(0.4135667662 * 2.99 / (x2/1000) , log(-log(y2 / 100)), 'b-', alpha=0.7, linewidth=0.8)
plt.xlabel('Energy (eV)')
plt.ylabel('ln(-ln(T))')

plt.show()
