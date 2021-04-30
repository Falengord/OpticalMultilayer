#support library for TMM simulations
#Paolo Wetzl 2020

import numpy as np
import matplotlib.pyplot as plt
import cmath

π = np.pi

#λ must be in micron.

#########################################################################################
#    Sellmeier equations for real part of the material's refractive indices             #
#########################################################################################

def n_sellmeier(λ,A,B,C):
    
    n2 = A
    for i in range(len(B)): 
        n2 = n2 + B[i]*λ**2/(λ**2-C[i])

    return np.sqrt(n2)

    
def material_nk_fnLN(λ, ray):
    '''
    cLN's ordinary and extraordinary refractive index (Zelmon 1997). 
    another ref. in Handbook of optical materials, much older (1972)
    Lambda in micron.
    '''
    A = 1

    B_ordinary      = [2.6734,  1.2290,  12.614]
    C_ordinary      = [0.01764, 0.05914, 474.60]

    B_extraordinary = [2.9804,  0.5981, 8.9543]
    C_extraordinary = [0.02047, 0.0666, 414.08]

    if   ray == 'o':
        B = B_ordinary.copy()
        C = C_ordinary.copy()
    elif ray == 'e':
        B = B_extraordinary.copy()
        C = C_extraordinary.copy()
    else:
        raise Exception('Ray polarization must be Ordinary(\'o\') or Extraordinary (\'e\')')

    return n_sellmeier(λ,A,B,C)



def material_nk_fnLNt(λ, temp, cLi, ray):
    '''
    LN's ordinary and extraordinary refractive index as a function of temperature (in °C) and Li composition in % 
    (Schlarb & Betzler 1994, https://journals.aps.org/prb/pdf/10.1103/PhysRevB.48.15613)
    Lambda in nm.
    '''
    λ = λ*1e3
    
    #f = f(T)-f(T_0) where T_0 is 24.5 °C
    f = (temp + 273)**2 + 4.0238e5*(1./np.tanh( 261.6/( temp + 273 ) ) - 1) - (24.5 + 273)**2 - 4.0238e5*(1./np.tanh( 261.6/( 24.5 + 273 ) ) - 1)     
    o = [4.5312e-5, 223.219, 2.7322e-5, 260.26, 3.6340e-8, 2.6613, 2.1203e-6, -1.827e-4]
    e = [3.9466e-5, 218.203, 8.3140e-5, 250.847, 3.0998e-8, 2.6613, 7.5187e-6, -3.8043e-5]

    if   ray == 'o':
        a = o.copy()
    elif ray == 'e':
        a = e.copy()
    else:
        raise Exception('Ray polarization must be Ordinary(\'o\') or Extraordinary (\'e\')')

    return np.sqrt((50.+cLi)/100. * a[0]/( 1./( a[1] + a[6]*f )**(2) - 1./λ**(2) ) + (50.-cLi)/100.  * a[2]/( 1./( a[3] + a[7]*f )**(2) - 1./λ**(2) ) - a[4]*λ**(2) + a[5] )



def material_nk_fnGaN(λ, ray):
    '''
    GaN's refractive index. Lambda in micron
    ref. Handbook of optical materials (refractiveindex.info)
    '''
      
    A_ordinary      = 3.6   
    B_ordinary      = [    1.75,      4.1]
    C_ordinary      = [0.256**2, 17.86**2]

    A_extraordinary = 5.35
    B_extraordinary = [    5.08,    1.0055]
    C_extraordinary = [18.76**2, 10.522**2]

    if   ray == 'o':
        A = A_ordinary
        B = B_ordinary.copy()
        C = C_ordinary.copy()
    elif ray == 'e':
        A = A_extraordinary
        B = B_extraordinary.copy()
        C = C_extraordinary.copy()
    else:
        raise Exception('Ray polarization must be Ordinary(\'o\') or Extraordinary (\'e\')')

    return n_sellmeier(λ,A,B,C)



def material_nk_fnAl2O3(λ, ray):
    '''
    Sapphire's refractive index. Lambda in micron.
    ref. Handbook of optical materials (refractiveindex.info)
    '''
    A = 1

    B_ordinary      = [   1.4313493,   0.65054713,    5.3414021]
    C_ordinary      = [0.0726631**2, 0.1193242**2, 18.028251**2]

    B_extraordinary = [1.5039759,      0.55069141,   6.59273791]
    C_extraordinary = [0.0740288**2, 0.1216529**2, 20.072248**2]

    if   ray == 'o':
        B = B_ordinary.copy()
        C = C_ordinary.copy()
    elif ray == 'e':
        B = B_extraordinary.copy()
        C = C_extraordinary.copy()
    else:
        raise Exception('Ray polarization must be Ordinary(\'o\') or Extraordinary (\'e\')')

    return n_sellmeier(λ,A,B,C)



#########################################################################################
#    Computation of asborption coefficients from various models                         #
#########################################################################################

def absorption(λ, alpha0, E0, Eu):
    '''
    INPUT: alpha0 and E0 are the Urbach parameters for the model, obtained from temperature dependent absorption measurements 
    (see Chichibu et a. 1997). Eu is the Urbach energy for the model. Lambda must be expressed in micron.
    DETAILS OF THE MODEL: single Urbach tail
    '''
    E = 0.4135667662 * 2.99 / λ 
    
    return alpha0 * np.exp((E - E0)/Eu)


def absorption_gauss(λ, EV, lambdaV, sigmaV, alphaV):
    '''
    INPUT: EV is an array containing various energies (Urbach energy, the E0 energy for the Urbach model).
    Similarly lambdaV contains wavelengths for the centering of Gaussian models of absorption,
    sigmaV for the width of such Gaussian curves and alphaV the constant absorption coefficients often used in such models.
    
    DETAILS OF THE MODEL: double Urbach tail with up to two possible Gaussian absorption curves (i.e. for simulating
    absorption related to impurity centers; for example Fe subsitutional impurity for Li)
    (not used in the final simulations)
    
    Lambda in micron
    Urbach and the other energies are in meV

    h = 4.135667662e-15 #eV s
    c = 2.99  #e8 ms^-1
    nu = c*1e14/λ
    '''

    E = 0.4135667662 * 2.99 / λ
    
    Eu  = EV[0]
    E0  = EV[1]
    Eu2 = EV[2]

    #lambda_fe = 0.531
    #lambda_fe = 0.58
    #sigma = 0.090

    lambda_fe = lambdaV[0]
    sigma_fe  = sigmaV[0]

    lambda2 = lambdaV[1]
    sigma2  = sigmaV[1]

    alpha0 = alphaV[0]
    alpha1 = alphaV[1]
    alpha2 = alphaV[2]
    alpha3 = alphaV[3]


    alpha_exp    = alpha0 * np.exp((E - E0) / (Eu))
    alpha_exp2   = alpha3 * np.exp((E - E0) / (Eu2))
    alpha_gauss1 = alpha1 * (1 / (sigma_fe * np.sqrt(2) * π)) * np.exp(-(λ - lambda_fe)**2 / (2 * sigma_fe**2))
    alpha_gauss2 = alpha2 * (1 / (sigma2 * np.sqrt(2) * π)) * np.exp(-(λ - lambda2)**2 / (2 * sigma_fe**2))


    alpha = alpha_exp + alpha_exp2 + alpha_gauss1 + alpha_gauss2

    return alpha
    
  
    
def absorption_other(λ, alpha0, lambda0, w0):
        
    return alpha0 * np.exp(-(λ - lambda0)**2 / (2*w0)**2)  



#########################################################################################################
#    Computation of the imaginary part of the refractive index from the absorption coefficient          #
#########################################################################################################

def make_k(λ, alpha0, E0, Eu):
    '''
    INPUT: alpha0 and E0 are the Urbach parameters for the model, obtained from temperature dependent absorption measurements 
    (see Chichibu et a. 1997). Eu is the Urbach energy for the model. Lambda must be expressed in micron.
    DETAILS OF THE MODEL: single Urbach tail
    '''
    a = absorption(λ, alpha0, E0, Eu)
       
    return a * λ / (4 * π) 


def make_k_gauss(λ, EV, lambdaV, sigmaV, alphaV):
    
    a = absorption_gauss(λ, EV, lambdaV, sigmaV, alphaV)
    
    return a * λ / (4 * π)

    
    
def make_k_other(λ, alpha0, lambda0, w0):
     
    a = absorption_other(λ, alpha0, lambda0, w0)
       
    return a * λ / (4 * π)    

    

#################################################
#    Transfer Matrix Method routines            #
#################################################    
    
#model of a 2x2 matrix as a 2D array (from TMM package in pip)      
def make_2x2_array(a, b, c, d, dtype=float):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]

    Same as "numpy.array([[a,b],[c,d]], dtype=float)", but ten times faster
    """
    my_array = np.empty((2,2), dtype=dtype)
    my_array[0,0] = a
    my_array[0,1] = b
    my_array[1,0] = c
    my_array[1,1] = d
    return my_array        
       
      
#computation of the Fresnel coefficients (reflection, transmission) for the interface (s polarization)

def fresnel_r(n_i,n_f,th,polarization='s'):
    
    th_i = np.arcsin(np.sin(th)/n_i.real)
    th_f = np.arcsin(np.sin(th)/n_f.real)
    
    if polarization == 's':
        return (n_i * np.cos(th_i) - n_f * np.cos(th_f))/(n_i * np.cos(th_i) + n_f * np.cos(th_f))        #s polarization
    elif polarization == 'p':
        return (n_i * np.cos(th_f) - n_f * np.cos(th_i))/(n_i * np.cos(th_f) + n_f * np.cos(th_i))        #p polarization
    else:
        raise Exception('Invalid polarization')
   
    
def fresnel_t(n_i,n_f,th,polarization='s'):
    
    th_i = np.arcsin(np.sin(th)/n_i.real)
    th_f = np.arcsin(np.sin(th)/n_f.real)

    if polarization == 's':
        return 2 * n_i * np.cos(th_i)/(n_i * np.cos(th_i) + n_f * np.cos(th_f))     #s polarization
    elif polarization == 'p':
        return 2 * n_i * np.cos(th_i)/(n_i * np.cos(th_f) + n_f * np.cos(th_i))     #p polarization
    else:
        raise Exception('Invalid polarization')
                
                               
#computation of the reflectivity/transmittivity reduction due to the interface roughness (see Katsidis)
#eq. 10 Katsidis
def rho(λ, z, n_i):
        
    return np.exp(-2 * (2*π*z * n_i / λ)**2 )



def tau(λ, z, n_i, n_f):

    return np.exp(-1./2 * (2*π*z * (n_f-n_i) / λ)**2 )


#computation of the transfer matrices for the interface and layer propagation, generic case (partial coherence due to interface roughness)
def make_layer_matr(λ, n_i, n_f, th, d, z):

    #eq 21 Katsidis
    
    #computation of the damping coefficients due to interface roughness (fw for forward propagating, bw for backward propagating)
    rho_fw  = rho(λ, z, n_i)
    rho_bw  = rho(λ, z, n_f)
    tau_fw  = tau(λ, z, n_i, n_f)

    th_i = np.arcsin(1. / n_i.real * np.sin(th))
    th_f = np.arcsin(1. / n_f.real * np.sin(th))

    #computation of the Fresnel coefficients at the interface, forward (m-1,m) and backwards (m,m-1)
    t_fw  = fresnel_t(n_i,n_f,th) * tau_fw
    t_bw  = fresnel_t(n_f,n_i,th) * tau_fw
    r_fw  = fresnel_r(n_i,n_f,th) * rho_fw
    r_bw  = fresnel_r(n_f,n_i,th) * rho_bw
    
    #computation of the coefficients due to the crossing of the layer
    p    = 2 * π /λ * n_i * d
    lay1 = np.exp( -1j*p)
    lay2 = np.exp( 1j*p)   
    
    return (1./t_fw) * np.array([lay1, -r_bw*lay1], [r_fw*lay2, (t_fw*t_bw - r_fw*r_bw) * lay2], dtype=complex)


#computation of the transfer matrices for the interface and layer propagation, starting from transfer matrices computed elsewhere
def T_inc(λ, T0m, TmN, n_f, d, th):

    dim = np.shape(λ)
    if dim == ():
        dim = 1

    #eq 16
    t0m = np.ones(dim)/T0m[0,0]
    tmN = np.ones(dim)/TmN[0,0]
    rm0 = -T0m[0,1]/T0m[0,0]
    rmN = TmN[1,0]/T0m[0,0]

    #from eq 14 to eq 16 (katsidis)
    th_f = np.arcsin(1 / n_f.real * np.sin(th))    
    arg  = 2 * π / λ * n_f * d * np.cos(th_f)
    
    return abs(t0m)**2 * abs(tmN)**2 / (abs(np.exp(1j*arg))**2 - abs(rm0*rmN)**2 * abs(np.exp(-1j*arg))**2 )


#computation of the transmission matrix for an interface. Basic approach       
def make_T_int(λ, n_i, n_f, th, z):

    dim = np.shape(λ)
    if dim == (): 
        dim = 1

    Tmix  = fresnel_r(n_i,n_f,th) * rho(λ, z, n_i)
    Tmix2 = fresnel_t(n_i,n_f,th) * tau(λ, z, n_i, n_f)
    T_int = [[np.ones(dim), Tmix], [Tmix, np.ones(dim)]]
    
    return np.array(T_int/Tmix2, dtype=complex)


                   
#computation of the transmission matrix for a layer. Basic approach      
def make_T_lay(λ, n, d, th):

    dim = np.shape(λ)
    if dim == (): 
        dim = 1

    th_f  = np.arcsin(1. / n.real * np.sin(th))
    p     = 2 * π /λ * n * d * np.cos(th_f)
    Tmix  = np.exp(-1j*p)
    Tmix2 = np.exp(1j*p)
    T_lay = [[Tmix, np.zeros(dim)], [np.zeros(dim), Tmix2]]

    return np.array(T_lay, dtype=complex)