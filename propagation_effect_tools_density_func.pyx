from __future__ import division
from global_vars import *
from libc.string cimport memcpy
from libc.stdlib cimport free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.math cimport sin, cos, tan, sqrt, exp, asin, acos, atan
from numpy cimport ndarray
from libcpp cimport bool
import os
from scipy.special import comb
from scipy.integrate import odeint
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
cimport cython
cimport numpy as np

#import propagation_effect_tools_initial_condition_cython as prop_tool_ini_cond

# cimport cython_wrapper_test


cdef double pi
pi = acos(-1.0)
np.import_array()

# density_func
cdef double normalize_phi(double phi):
    cdef double abs_phi,phi1
    abs_phi	=abs(phi)
    phi1	=np.sign(phi)*(abs_phi-2*pi*int(abs_phi/(2*pi)))
    if phi1 < 0:
        phi1+=2*pi
    return phi1

cdef double normalize_theta(double theta):
    cdef double abs_theta,theta1
    abs_theta	=abs(theta)
    theta1	=np.sign(theta)*(abs_theta-2*pi*int(abs_theta/(2*pi)))
    if theta1 < 0:
        theta1+=2*pi
    if theta1 > pi:
        theta1=2*pi-theta1
    return theta1

cpdef double density_func(double n_p0, double r, double theta, double phi, float r_max, float rho_fact):  # already cythonized
    cdef double rho,val
    try:
        val =  interp_rho([normalize_phi(phi), r, normalize_theta(theta)])[0]
        if np.isnan(val)==True:
            val=0.
        rho = n_p0 * ((1. / r) + (rho_fact *val))
    except ValueError:
        print(phi,normalize_phi(phi),normalize_theta(theta),r,'durrrrrrrr')
    #rho = n_p0 * ((1. / r) + (rho_fact * interp_rho([normalize_phi(phi), r, normalize_theta(theta)])[0]))
    #print(rho)
    return rho

# already cythonized
cpdef double delta_n_delta_theta(double n_p0, double r, double theta, double phi, float r_max, float rho_fact):
    cdef double dn_dtheta,val
    val =interp_dn_dtheta([normalize_phi(phi), r, normalize_theta(theta)])[0]
    if np.isnan(val)==True:
        val =0.0
    dn_dtheta = n_p0 * rho_fact * val
    return dn_dtheta

# already cythonized
cpdef double delta_n_delta_phi(double n_p0, double r, double theta, double phi, float r_max, float rho_fact):
    cdef double dn_dphi,val
    val     = interp_dn_dphi([normalize_phi(phi), r, normalize_theta(theta)])[0]
    if np.isnan(val)==True:
        val =0.0
    dn_dphi = n_p0 * rho_fact * val
    return dn_dphi

# already cythonized
cpdef double delta_n_delta_r(double n_p0, double r, double theta, double phi, float r_max, float rho_fact):
    cdef double dn_dr,val
    val     = interp_dn_dr([normalize_phi(phi), r, normalize_theta(theta)])[0]
    if np.isnan(val)==True:
        val =0.0
    dn_dr   = n_p0 * ((-1. / r**2) + (rho_fact *
                   val ))
    return dn_dr
