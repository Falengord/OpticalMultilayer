#support library for TMM simulations
##
#Paolo Wetzl 2020

from __future__ import division, print_function, absolute_import

from numpy import pi, array, sqrt, exp, cos, empty, imag, tanh
import matplotlib.pyplot as plt
   



#########################################################################################
#	Sellmeier equations for real part of the material's refractive indices		#
#########################################################################################

def sellmeier(a,lambda_vac):

	#beware of which coefficients are to be squared or not
	lamb = lambda_vac**2
	
	return sqrt(a[0] + a[1] * lamb / (lamb - a[2]) + a[3] * lamb /
	(lamb - a[4]) + a[5] * lamb / (lamb - a[6]))
    
    
    
def material_nk_fnLN(lambda_vac, ray):
	
	#cLN's ordinary and extraordinary refractive index (Zelmon 1997). Lambda must be expressed in microm
	#another ref. in Handbook of optical materials, much older (1972)
	
	lamb = lambda_vac**2

	o = [1, 2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60]
	e = [1, 2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 414.08]
	
	if ray == 'o':
		a = o.copy()
		
	else:
		a = e.copy()	

	return sellmeier(a,lambda_vac)
	
	

def material_nk_fnLNt(lambda_vac, temp, cLi, ray):
	
	#LN's ordinary and extraordinary refractive index as a function of temperature (in C°) and Li composition in % (Schlarb & Betzler 1994). Coeff. are for lambda in nm, while lambda_vac is in microm
	
	#from microm to nm for the calculation
	lambda_vac = lambda_vac*1e3
	lamb = lambda_vac**2	
	
	f = (temp + 273)**2 + 4.0238e5*(1./tanh( 261.6/( temp + 273 ) ) - 1) - (24.5 + 273)**2 + 4.0238e5*(1./tanh( 261.6/( 24.5 + 273 ) ) - 1)

	o = [ 4.5312e-5 , 223.219, 2.7322e-5, 260.26, 3.6340e-8, 2.6613, 2.1203e-6, -1.827e-4]
	e = [ 3.9466e-5 , 218.203, 8.3140e-5, 250.847, 3.0998e-8, 2.6613, 7.5187e-6, -3.8043e-5]
	
	if ray == 'o':
		a = o.copy()
		
	else:
		a = e.copy()	

	return sqrt( (50+cLi)/100. * a[0]/( 1./( a[1] + a[6]*f )**(2) - 1./lamb ) + (50-cLi)/100.  * a[2]/( 1./( a[3] + a[7]*f )**(2) - 1./lamb ) - a[4]*lamb + a[5] )
	


def material_nk_fnGaN(lambda_vac, ray):

	#GaN's refractive index
	#lambda in microm
	#ref. Handbook of optical materials (refractiveindex.info)
	o = [3.6, 1.75, 0.256**2, 4.1, 17.86**2, 0, 0]
	e = [5.35, 5.08, 18.76**2, 1.0055, 10.522**2, 0, 0]

	#n = sqrt(3.6 + 1.75 * lamb / (lamb - 0.256**2) + 4.1 * lamb /
	#         (lamb - 17.86**2))


	if ray == 'o':
		a = o.copy()
		
	else:
		a = e.copy()	

	return sellmeier(a,lambda_vac)
	


def material_nk_fnAl2O3(lambda_vac, ray):

	#sapphire's refractive index
	#lambda in microm
	#ref. Handbook of optical materials (refractiveindex.info)
	o = [1, 1.4313493, 0.0726631**2, 0.65054713, 0.1193242**2, 5.3414021, 18.028251**2]
	e = [1, 1.5039759, 0.0740288**2, 0.55069141, 0.1216529**2, 6.59273791, 20.072248**2]

	#n = sqrt(1 + 1.4313493 * lamb / (lamb - 0.0726631**2) + 0.65054713 * lamb /
	#     (lamb - 0.1193242**2) + 5.3414021 * lamb / (lamb - 18.028251**2))

	if ray == 'o':
		a = o.copy()
		
	else:
		a = e.copy()	

	return sellmeier(a,lambda_vac)
	
	
	
#########################################################################################
#	Computation of asborption coefficients from various models			#
#########################################################################################

def absorption(lambda_vac, alpha0, E0, Eu):

	#INPUT: alpha0 and E0 are the Urbach parameters for the model, obtained from temperature dependent absorption measurements (see Chichibu et a. 1997). Eu is the Urbach energy for the model.

	#DETAILS OF THE MODEL: single Urbach tail

	E = 0.4135667662 * 2.99 / lambda_vac 	
	    
	return alpha0 * exp((E - E0)/Eu )  
	


def absorption_gauss(lambda_vac, EV, lambdaV, sigmaV, alphaV):

	#INPUT: EV is an array containing various energies (Urbach energy, the E0 energy for the Urbach model).
	#Similarly lambdaV contains wavelengths for the centering of Gaussian models of absorption,
	#sigmaV for the width of such Gaussian curves and alphaV the constant absorption coefficients often used in such models.
	
	#DETAILS OF THE MODEL: double Urbach tail with up to two possible Gaussian absorption curves (i.e. for simulating
	#absorption related to impurity centers; for example Fe subsitutional impurity for Li)
	#(not used in the final simulations)
	
	#lambda in microm
    	#Urbach and the other energies are in milli eV

   	#h = 4.135667662e-15 #eV s
    	#c = 2.99  #e8 ms^-1
    	#nu = c*1e14/lambda_vac
	E = 0.4135667662 * 2.99 / lambda_vac
	
	Eu = EV[0]
	E0 = EV[1]
	Eu2 = EV[2]

	#lambda_fe = 0.531
	#lambda_fe = 0.58
	#sigma = 0.090

	lambda_fe = lambdaV[0]
	sigma_fe = sigmaV[0]

	lambda2 = lambdaV[1]
	sigma2 = sigmaV[1]

	alpha0 = alphaV[0]
	alpha1 = alphaV[1]
	alpha2 = alphaV[2]
	alpha3 = alphaV[3]


	alpha_exp = alpha0 * exp((E - E0) / (Eu))

	alpha_exp2 = alpha3 * exp((E - E0) / (Eu2))

	alpha_gauss1 = alpha1 * (1 / (sigma_fe * sqrt(2) * pi)) * exp(
	-(lambda_vac - lambda_fe)**2 / (2 * sigma_fe**2))
	alpha_gauss2 = alpha2 * (1 / (sigma2 * sqrt(2) * pi)) * exp(
	-(lambda_vac - lambda2)**2 / (2 * sigma_fe**2))


	alpha = alpha_exp + alpha_exp2 + alpha_gauss1 + alpha_gauss2

	return alpha
    
  
    
def absorption_other(lambda_vac, alpha0, lambda0, w0):
	    
	return alpha0 * exp(-(lambda_vac - lambda0)**2 / (2*w0)**2)	        



#########################################################################################################
#	Computation of the imaginary part of the refractive index from the absorption coefficient	#
#########################################################################################################

def make_k(lambda_vac, alpha0, E0, Eu):
  
    a = absorption(lambda_vac, alpha0, E0, Eu)
       
    return a * lambda_vac / (4 * pi)   



def make_k_gauss(lambda_vac, EV, lambdaV, sigmaV, alphaV):
  
    a = absorption_gauss(lambda_vac, EV, lambdaV, sigmaV, alphaV)
    return a * lambda_vac / (4 * pi)

    
    
def make_k_other(lambda_vac, alpha0, lambda0, w0):
  
    a = absorption_other(lambda_vac, alpha0, lambda0, w0)
       
    return a * lambda_vac / (4 * pi)    
    
    

#################################################
#	Transfer Matrix Method routines		#
#################################################    
    
#model of a 2x2 matrix as a 2D array (from TMM package in pip)      
def make_2x2_array(a, b, c, d, dtype=float):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]

    Same as "numpy.array([[a,b],[c,d]], dtype=float)", but ten times faster
    """
    my_array = empty((2,2), dtype=dtype)
    my_array[0,0] = a
    my_array[0,1] = b
    my_array[1,0] = c
    my_array[1,1] = d
    return my_array        
       
      
       
#computation of the Fresnel coefficients (reflection, transmission) for the interface
def fresnel_r(n_i,n_f,th_i,th_f):

	return (n_i * cos(th_i) - n_f * cos(th_f))/(n_i * cos(th_i) + n_f * cos(th_f))
	
	       
	       
def fresnel_t(n_i,n_f,th_i,th_f):

	return 2 * n_i * cos(th_i)/(n_i * cos(th_i) + n_f * cos(th_f)) 
                
                
                
#computation of the reflectivity/transmittivity reduction due to the interface roughness (see Katsidis)                        
def rho(lambda_vac, z, n_i):

	return exp(-2 * (2*pi*z * n_i / lambda_vac)**2 )
	
	
	
def tau(lambda_vac, z, n_i, n_f):

	return exp(-1./2 * (2*pi*z * (n_f-n_i) / lambda_vac)**2 )



#computation of the transfer matrices for the interface and layer propagation, generic case (partial coherence due to interface roughness)
def make_layer_matr(lambda_vac, n_i, n_f, th_i, th_f, d, z):

	#eq 21 Katsidis
	
	#computation of the damping coefficients due to interface roughness (fw for forward propagating, bw for backward propagating)
	rho_fw  = rho(lambda_vac, z, n_i)
	rho_bw  = rho(lambda_vac, z, n_f)
	tau_fw  = tau(lambda_vac, z, n_i, n_f)

	#computation of the Fresnel coefficients at the interface, forward (m-1,m) and backwards (m,m-1)
	t_fw  = fresnel_t(n_i,n_f,th_i,th_f) * tau_fw
	t_bw  = fresnel_t(n_f,n_i,th_f,th_i) * tau_fw
	r_fw  = fresnel_r(n_i,n_f,th_i,th_f) * rho_fw
	r_bw  = fresnel_r(n_f,n_i,th_f,th_i) * rho_bw
	
	#computation of the coefficients due to the crossing of the layer
	p = 2 * pi /lambda_vac * n_i * d
	lay1 = exp( -1j*p)
	lay2 = exp( 1j*p)	
	
	return (1./t_fw) * make_2x2_array(lay1, -r_bw*lay1, r_fw*lay2, ( t_fw*t_bw - r_fw*r_bw) * lay2, dtype=complex)



#computation of the transfer matrices for the interface and layer propagation, starting from transfer matrices computed elsewhere
def T_inc(lambda_vac, T0m, TmN, n_f, d):

	#implemento eq 16	
	t0m = 1./T0m[0,0]
	tmN = 1./TmN[0,0]
	rm0 = -T0m[0,1]/T0m[0,0]
	rmN = TmN[1,0]/T0m[0,0]
	
	#old: eq 16 (katsidis)
	#arg = (pi/lambda_vac)*imag(n_f)*d
	#return abs(t0m)**2 * (abs(tmN)**2 * exp( -2*arg ) ) / (1 - abs(rm0*rmN)**2 * exp( -4*arg ) )
	
	#new: from eq 14 to eq 16 (katsidis)
	arg = 2 * pi /lambda_vac * n_f * d
	
	return abs(t0m)**2 * abs(tmN)**2 / (abs(exp(1j*arg))**2 - abs(rm0*rmN)**2 * abs(exp(-1j*arg))**2 )



#computation of the transmission matrix for an interface. Basic approach	        
def make_T_int(lambda_vac, n_i, n_f, th_i, th_f, z):	

	Tmix = fresnel_r(n_i,n_f,th_i,th_f) * rho(lambda_vac, z, n_i)
	Tmix2 = fresnel_t(n_i,n_f,th_i,th_f) * tau(lambda_vac, z, n_i, n_f) 
		
	return (1./Tmix2) * make_2x2_array(1, Tmix, Tmix, 1, dtype=complex)

        
           
#computation of the transmission matrix for a layer. Basic approach	        
def make_T_lay(lambda_vac, n, d):	

	p = 2 * pi /lambda_vac * n * d

	Tmix = exp( -1j*p)
	Tmix2 = exp( 1j*p)
	
	return make_2x2_array(Tmix, 0, 0, Tmix2, dtype=complex)
 
