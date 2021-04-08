#Paolo Wetzl 2020

#IMPORT LIBRERIE
from numpy import linspace, array, loadtxt, exp, dot, log, log10, savetxt, transpose
import matplotlib.pyplot as plt
    
from fun import material_nk_fnLN, material_nk_fnLNt, material_nk_fnGaN, material_nk_fnAl2O3, make_T_int, make_T_lay, make_k, make_layer_matr, T_inc

#layer characteristics

#1: air
#2: LN
#3: GaN
#4: Al2O3
#5: air 


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
lambda_list = linspace(l_sx, l_dx, n_points)


#initialization of T matrices
T = []
Tinc = []
	
	
#initialization of T matrices	
th_i = 0
th_f = 0


#execute the various operations for each wavelength
for lambda_vac in lambda_list:

	#compute the refractive indeces of the various materials
	nreGaN = material_nk_fnGaN(lambda_vac, 'o')
	nimGaN = make_k(lambda_vac, alpha0_GaN, E0_GaN, Eu_GaN)
	
	nreLN = material_nk_fnLN(lambda_vac,'o')
	nimLN = make_k(lambda_vac, alpha0_LN, E0_LN, Eu_LN)

	n_LN = nreLN + 1j * nimLN    
	n_GaN = nreGaN + 1j * nimGaN
	n_Al2O3 = material_nk_fnAl2O3(lambda_vac, 'o')

	#compute the transfer matrices for the various interfaces and layers, one by one
	
	T12 = make_T_int(lambda_vac, 1, n_LN, th_i, th_f, z_1)
	T2  = make_T_lay(lambda_vac, n_LN, d_LN)
	T23 = make_T_int(lambda_vac, n_LN, n_GaN, th_i, th_f, z_2)	
	T3  = make_T_lay(lambda_vac, n_GaN, d_GaN)	
	T34 = make_T_int(lambda_vac, n_GaN, n_Al2O3, th_i, th_f, z_3)
	T4  = make_T_lay(lambda_vac, n_Al2O3, d_Al2O3)			
	T41 = make_T_int(lambda_vac, n_Al2O3, 1, th_i, th_f, z_4)
	
	#use eq 16 (Katsidis) for the computation of the transmission coefficient of the multilayer from the total transfer matrix for the stack of layers
	Tinc.append(T_inc(lambda_vac, dot(dot(dot(dot(dot(T12,T2),T23),T3),T34),T4), T41, n_Al2O3, d_Al2O3))



#load the measurement datafile to compare it with the simulation
x, y = loadtxt('Dati/7.2 con bandwidth bassa.txt', delimiter='\t', unpack=True)



#save the result of the simulation on a file
d =array([lambda_list,Tinc]).T
savetxt('simLN_7.2.txt', d, fmt='%10.5f', newline='\n')



#plot the simulation and the measured data in the same graph
plt.figure(1)
plt.plot(0.4135667662 * 2.99 / lambda_list, Tinc, 'g-', alpha=0.7, linewidth=0.8)
plt.plot(0.4135667662 * 2.99 / (x/1000), y / 100, 'b-', alpha=0.5, linewidth=0.8)
plt.xlabel('Energy (eV)')
plt.ylabel('Fraction of power transmitted')

plt.title('Transmission at normal incidence')
plt.savefig('Transmission_7.2.pdf')
plt.savefig('Transmission_7.2.svg')


#plot the data in order to be able to check the slope of the absorption edge (to estimate the Urbach energy)
plt.figure(2)
plt.xlim(2.6, 3.5)
plt.plot(0.4135667662 * 2.99 / lambda_list , log(-log(Tinc)), 'g-', alpha=0.7, linewidth=0.8)
plt.plot(0.4135667662 * 2.99 / (x/1000) , log(-log(y / 100)), 'b-', alpha=0.7, linewidth=0.8)
plt.xlabel('Energy (eV)')
plt.ylabel('ln(-ln(T))')

plt.show()
