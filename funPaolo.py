#support library for TMM simulations
#Paolo Wetzl 2020

import numpy as np
import matplotlib.pyplot as plt


π = np.pi

#########################################################################################
#    Sellmeier equations for real part of the material's refractive indices                #
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
    '''
    λ = λ*1e-3
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
    (Schlarb & Betzler 1994).
    https://journals.aps.org/prb/pdf/10.1103/PhysRevB.48.15613
    '''

    #f(T)-f(T_0) where T_0 is 24.5 °C
    f = (temp + 273)**2 + 4.0238e5*(1./tanh( 261.6/( temp + 273 ) ) - 1) - (24.5 + 273)**2 - 4.0238e5*(1./tanh( 261.6/( 24.5 + 273 ) ) - 1)     
    o = [4.5312e-5, 223.219, 2.7322e-5, 260.26, 3.6340e-8, 2.6613, 2.1203e-6, -1.827e-4]
    e = [3.9466e-5, 218.203, 8.3140e-5, 250.847, 3.0998e-8, 2.6613, 7.5187e-6, -3.8043e-5]

    if   ray == 'o':
        a = o.copy()
    elif ray == 'e':
        a = e.copy()
    else:
        raise Exception('Ray polarization must be Ordinary(\'o\') or Extraordinary (\'e\')')

    return sqrt((50+cLi)/100. * a[0]/( 1./( a[1] + a[6]*f )**(2) - 1./lamb ) + (50-cLi)/100.  * a[2]/( 1./( a[3] + a[7]*f )**(2) - 1./lamb ) - a[4]*lamb + a[5] )



def material_nk_fnGaN(λ, ray):
    '''
    GaN's refractive index. Lambda in microm
    ref. Handbook of optical materials (refractiveindex.info)
    '''
    λ = λ*1e-3
      
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
    Sapphire's refractive index. Lambda in micron
    ref. Handbook of optical materials (refractiveindex.info)
    '''
    λ = λ*1e-3
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
#    Computation of asborption coefficients from various models                            #
#########################################################################################

def absorption(λ, alpha0, E0, Eu):
    '''
    INPUT: alpha0 and E0 are the Urbach parameters for the model, obtained from temperature dependent absorption measurements 
    (see Chichibu et a. 1997). Eu is the Urbach energy for the model. Lambda must be expressed in micron
    DETAILS OF THE MODEL: single Urbach tail
    '''
    λ = λ*1e-3
    E = 0.4135667662 * 2.99 / λ 
    
    return alpha0 * np.exp((E - E0)/Eu)  



#########################################################################################################
#    Computation of the imaginary part of the refractive index from the absorption coefficient            #
#########################################################################################################

def make_k(λ, alpha0, E0, Eu):
    '''
    INPUT: alpha0 and E0 are the Urbach parameters for the model, obtained from temperature dependent absorption measurements 
    (see Chichibu et a. 1997). Eu is the Urbach energy for the model. Lambda must be expressed in micron
    DETAILS OF THE MODEL: single Urbach tail
    '''
    a = absorption(λ, alpha0, E0, Eu)
       
    return a * λ / (4 * π)     
    

#################################################
#    Transfer Matrix Method routines                #
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

def fresnel_r(n_i,n_f,th):
    
    th_i = np.sinh(1. / n_i * np.sin (th))
    th_f = np.sinh(1. / n_f * np.sin (th))
    
    return (n_i * np.cos(th_i) - n_f * np.cos(th_f))/(n_i * np.cos(th_i) + n_f * np.cos(th_f))        #s polarization
    #return (n_i * cos(th_f) - n_f * cos(th_i))/(n_i * cos(th_f) + n_f * cos(th_i))       #p polarization
   
    
def fresnel_t(n_i,n_f,th):
    
    th_i = np.sinh(1. / n_i * np.sin (th))
    th_f = np.sinh(1. / n_f * np.sin (th))

    return 2 * n_i * np.cos(th_i)/(n_i * np.cos(th_i) + n_f * np.cos(th_f))     #s polarization
    #return 2 * n_i * cos(th_i)/(n_i * cos(th_f) + n_f * cos(th_i))    #p polarization
                
                               
#computation of the reflectivity/transmittivity reduction due to the interface roughness (see Katsidis)
#eq. 10 Katsidis
def rho(λ, z, n_i):
    return np.exp(-2 * (2*π*z * n_i / λ)**2 )



def tau(λ, z, n_i, n_f):
    return np.exp(-1./2 * (2*π*z * (n_f-n_i) / λ)**2 )


#computation of the transfer matrices for the interface and layer propagation, starting from transfer matrices computed elsewhere
def T_inc(λ, T0m, TmN, n_f, d, th):

    #implemento eq 16
    t0m = 1./T0m[0,0]
    tmN = 1./TmN[0,0]
    rm0 = -T0m[0,1]/T0m[0,0]
    rmN = TmN[1,0]/T0m[0,0]

    th_f = sinh(1. / n_f * sin (th))    
    arg = 2 * pi /λ * n_f * d * cos(th_f)
    
    return abs(t0m)**2 * abs(tmN)**2 / (abs(exp(1j*arg))**2 - abs(rm0*rmN)**2 * abs(exp(-1j*arg))**2 )


#computation of the transmission matrix for an interface. Basic approach       
def make_T_int(λ, n_i, n_f, th, z):
    Tmix  = fresnel_r(n_i,n_f,th) * rho(λ, z, n_i)
    Tmix2 = fresnel_t(n_i,n_f,th) * tau(λ, z, n_i, n_f)
    
    return (1./Tmix2) * np.array([[1, Tmix], [Tmix, 1]], dtype=complex)


                   
#computation of the transmission matrix for a layer. Basic approach      
def make_T_lay(λ, n, d, th):  
    th_f  = np.sinh(1. / n * np.sin (th))
    p     = 2 * π /λ * n * d * np.cos(th_f)
    Tmix  = np.exp(-1j*p)
    Tmix2 = np.exp(1j*p)
        
    return np.array([[Tmix, 0], [0, Tmix2]], dtype=complex)