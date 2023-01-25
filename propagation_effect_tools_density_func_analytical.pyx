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

'''
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


'''


cpdef double density_func(double n_p0,double r,double theta,double phi,float r_max, float rho_fact): #already cythonized
    cdef double theta0, r0, y, z, x, sigma, factor, n_e
    cdef float M
    theta0	=(pi/2.0)*(1.+sin(phi))
    r0	=2.5
    M	=5.
    y	=1./(1.+exp(2.*M*(r-r0)))
    z	=r*sin(theta-theta0)
    x	=r*cos(theta-theta0)
    sigma	=0.7*exp(r_max/(x**2.+r_max))
    factor	=1.5*2	# To fiddle with the extent of the density enhancement from the magnetic equator along the Z-axis
    n_e	=rho_fact*(n_p0/r)*exp(-(factor*z**2)/sigma**2)*(1-y)+n_p0/r
    return n_e

cpdef double delta_n_delta_theta(double n_p0,double r,double theta,double phi,float r_max, float rho_fact): #already cythonized
    cdef double theta0, r0, y, z, x, sigma, factor, dz_dt, dx_dt, dsigma_dt, dn_dt
    cdef float M=5.0
    theta0	=(pi/2.0)*(1+sin(phi))
    r0	=2.5
    y	=1./(1.+exp(2*M*(r-r0)))
    z	=r*sin(theta-theta0)
    x	=r*cos(theta-theta0)
    sigma	=0.7*exp(r_max/(x**2.+r_max))
    factor	=1.5*2.	
    dz_dt	=x
    dx_dt	=-z
    dsigma_dt=(-2.*sigma*r_max*x*dx_dt)/(x**2.+r_max)**2.
    dn_dt	=-2*rho_fact*(n_p0/r)*(1-y)*exp(-(factor*z**2)/sigma**2)*factor*(((z*dz_dt)/sigma**2)-(z**2/sigma**3)*dsigma_dt)
    return dn_dt

cpdef double delta_n_delta_phi(double n_p0,double r,double theta,double phi,float r_max, float rho_fact): #already cythonized
    cdef double theta0, dtheta0_dphi, r0, y, z, x, sigma, factor, dz_dphi, dx_dphi, dsigma_dphi, dn_dphi
    cdef float M=5.0
    theta0	=(pi/2.0)*(1.+sin(phi))
    dtheta0_dphi=(pi/2.0)*cos(phi)
    r0	=2.5
    y	=1/(1+exp(2*M*(r-r0)))
    z	=r*sin(theta-theta0)
    x	=r*cos(theta-theta0)
    sigma	=0.7*exp(r_max/(x**2+r_max))
    factor	=1.5*2.
    dz_dphi	=-x*dtheta0_dphi
    dx_dphi	=z*dtheta0_dphi
    dsigma_dphi=(-2.*sigma*r_max*x*dx_dphi)/(x**2+r_max)**2
    dn_dphi	=-2*rho_fact*(n_p0/r)*(1-y)*exp(-(factor*z**2)/sigma**2)*factor*(((z*dz_dphi)/sigma**2)-(z**2/sigma**3)*dsigma_dphi)
    return dn_dphi

cpdef double delta_n_delta_r(double n_p0,double r,double theta,double phi,float r_max, float rho_fact): #already cythonized
    cdef double theta0, r0, y, z, x, sigma, factor, dy_dr, dz_dr, dx_dr, dsigma_dr, n_e, part1, part2
    cdef float M=5.0
    theta0	=(pi/2.0)*(1+sin(phi))
    r0	=2.5
    y	=1./(1.+exp(2*M*(r-r0)))
    z	=r*sin(theta-theta0)
    x	=r*cos(theta-theta0)
    sigma	=0.7*exp(r_max/(x**2.+r_max))
    factor	=1.5*2.
    dy_dr	=-2*M*y**2*exp(2*M*(r-r0))
    dx_dr	=cos(theta-theta0)
    dz_dr	=sin(theta-theta0)
    dsigma_dr=-2*x*sigma*(r_max/(x**2.+r_max)**2)*dx_dr
    n_e	=rho_fact*(n_p0/r)*exp(-(factor*z**2)/sigma**2)*(1-y)+n_p0/r
    part1	=(-n_p0/r**2)*(1+rho_fact*exp(-(factor*z**2)/sigma**2)*(1-y))
    part2	=-(n_p0/r)*rho_fact*exp(-(factor*z**2)/sigma**2)*(dy_dr+2*factor*(1-y)*(z/sigma)*((dz_dr/sigma)-(z/sigma**2)*dsigma_dr))
    return part1+part2

    
