from __future__ import division
cimport cython
cimport numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import odeint
from scipy.special import comb
import os
#import propagation_effect_tools_initial_condition_3D_complex_IM_cython as prop_tool_ini_cond
import propagation_effect_tools_density_func as prop_tool_density
from libcpp cimport bool
from numpy cimport ndarray
from libc.math cimport sin, cos, tan, sqrt, exp, asin, acos, atan
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memcpy 

cdef double pi
pi	=acos(-1.0)

cdef double get_plasma_freq(double n_p0,double r,double theta,double phi,float r_max,float rho_fact):  #already cythonized
	cdef double n_e=prop_tool_density.density_func(n_p0,r,theta,phi,r_max,rho_fact) #in cgs
	return 9.0*10.**(-3)*sqrt(n_e) # in MHz
'''
cpdef double density_func(double n_p0,double r,double theta,double phi,float r_max): #already cythonized
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
	n_e	=0*100*(n_p0/r)*exp(-(factor*z**2)/sigma**2)*(1-y)+n_p0/r
	return n_e
cdef double delta_n_delta_theta(double n_p0,double r,double theta,double phi,float r_max): #already cythonized
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
	dn_dt	=-2*0*100*(n_p0/r)*(1-y)*exp(-(factor*z**2)/sigma**2)*factor*(((z*dz_dt)/sigma**2)-(z**2/sigma**3)*dsigma_dt)
	return dn_dt
cdef double delta_n_delta_phi(double n_p0,double r,double theta,double phi,float r_max): #already cythonized
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
	dn_dphi	=-2*0*100*(n_p0/r)*(1-y)*exp(-(factor*z**2)/sigma**2)*factor*(((z*dz_dphi)/sigma**2)-(z**2/sigma**3)*dsigma_dphi)
	return dn_dphi
cdef double delta_n_delta_r(double n_p0,double r,double theta,double phi,float r_max): #already cythonized
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
	n_e	=0*100*(n_p0/r)*exp(-(factor*z**2)/sigma**2)*(1-y)+n_p0/r
	part1	=(-n_p0/r**2)*(1+0*100*exp(-(factor*z**2)/sigma**2)*(1-y))
	part2	=-(n_p0/r)*0*100*exp(-(factor*z**2)/sigma**2)*(dy_dr+2*factor*(1-y)*(z/sigma)*((dz_dr/sigma)-(z/sigma**2)*dsigma_dr))
	return part1+part2	
'''
cdef delta_cos_sq_psi_delta_theta(double cos_psi,double r,double theta,double theta_dash,double phi_dash):#alternate function written
	cdef double k, B, dot_kB, d_dot_kB_dt, dk_dt, dB_dt
	cdef float Bp
	cdef np.ndarray[np.double_t, ndim=1] k_vect_volar
	cdef np.ndarray[np.double_t, ndim=1] B_vect_polar
	Bp			=1.0
	k_vect_polar		=np.array([1,r*theta_dash,r*sin(theta)*phi_dash])
	B_vect_polar		=B_func(r,theta,Bp)
	k,B			=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	dot_kB			=np.dot(k_vect_polar,B_vect_polar)
	d_dot_kB_dt		=delta_dot_kB_delta_theta(r,theta,theta_dash,Bp)
	dk_dt			=delta_k_delta_theta(k,r,theta,theta_dash,phi_dash)
	dB_dt			=delta_B_delta_theta(r,theta,Bp)
	return 2.*cos_psi**2*((d_dot_kB_dt/dot_kB)-(dk_dt/k)-(dB_dt/B))


cdef delta_cos_sq_psi_delta_phi(double cos_psi,double r,double theta,double theta_dash,double phi_dash):#alternate function written
	cdef float Bp
	cdef double k, B, dot_kB, d_dot_kB_dphi, dk_dphi, dB_dphi
	cdef np.ndarray[np.double_t, ndim=1] k_vect_volar
	cdef np.ndarray[np.double_t, ndim=1] B_vect_polar
	Bp			=1.0
	k_vect_polar		=np.array([1,r*theta_dash,r*sin(theta)*phi_dash])
	B_vect_polar		=B_func(r,theta,Bp)
	k,B			=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	dot_kB			=np.dot(k_vect_polar,B_vect_polar)
	d_dot_kB_dphi		=0.0
	dk_dphi			=0.0
	dB_dphi			=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dphi/dot_kB)-(dk_dphi/k)-(dB_dphi/B))

cdef delta_cos_sq_psi_delta_theta_dash(double cos_psi,double r,double theta,double theta_dash,double phi_dash):#alternate function written
	cdef float Bp
	cdef double k, B, dot_kB, d_dot_kB_dt_dash, dk_dt_dash, dB_dt_dash
	cdef np.ndarray[np.double_t, ndim=1] k_vect_volar
	cdef np.ndarray[np.double_t, ndim=1] B_vect_polar
	Bp			=1.0
	k_vect_polar		=np.array([1,r*theta_dash,r*sin(theta)*phi_dash])
	B_vect_polar		=B_func(r,theta,Bp)
	k,B			=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	dot_kB			=np.dot(k_vect_polar,B_vect_polar)
	d_dot_kB_dt_dash	=delta_dot_kB_delta_theta_dash(r,theta,theta_dash,Bp)
	dk_dt_dash		=delta_k_delta_theta_dash(k,r,theta,theta_dash,phi_dash)
	dB_dt_dash		=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dt_dash/dot_kB)-(dk_dt_dash/k)-(dB_dt_dash/B))

cdef delta_cos_sq_psi_delta_phi_dash(double cos_psi,double r,double theta,double theta_dash,double phi_dash):#alternate function written
	cdef double k, B, dot_kB, d_dot_kB_dphi_dash, dk_dphi_dash, dB_dphi_dash
	cdef float Bp
	cdef np.ndarray[np.double_t, ndim=1] k_vect_volar
	cdef np.ndarray[np.double_t, ndim=1] B_vect_polar
	Bp			=1.0
	k_vect_polar		=np.array([1,r*theta_dash,r*sin(theta)*phi_dash])
	B_vect_polar		=B_func(r,theta,Bp)
	k,B			=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	dot_kB			=np.dot(k_vect_polar,B_vect_polar)
	d_dot_kB_dphi_dash	=0.0
	dk_dphi_dash		=delta_k_delta_phi_dash(k,r,theta,theta_dash,phi_dash)
	dB_dphi_dash		=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dphi_dash/dot_kB)-(dk_dphi_dash/k)-(dB_dphi_dash/B))


cdef B_func(double r,double theta,float Bp):#alternate function written
	cdef double Br, Bt
	Br,Bt=(cos(theta))/r**3,(0.5*sin(theta))/r**3
	return np.array([Br*Bp, Bt*Bp,0])

cdef get_vector_modulus(np.ndarray[np.double_t,ndim=1] V):#alternate function written
	return sqrt(np.sum(V**2))

cdef double get_sigma(float nu,double nu_p,double nu_B,double cos_psi): #already cythonized
	cdef double factor
	factor		=0.5*nu/abs(nu**2-nu_p**2)
	return 	factor*nu_B*(1-cos_psi**2)

cdef double get_tau(double sigma,double cos_psi,double nu_p,float nu): #already cythonized
	if nu>nu_p:
		return -(sigma+sqrt(sigma**2+cos_psi**2))
	else:
		return (sigma+sqrt(sigma**2+cos_psi**2))

cdef double get_refractive_index(double nu_p,double nu_B,double tau,double cos_psi,float nu,int mode):  #already cythonized
	if mode==1:	# X-mode
		return sqrt(1-(nu_p**2/(nu*(nu+tau*nu_B))))
	else:		# O-mode
		return sqrt(1-((tau*nu_p**2)/(nu*(tau*nu-nu_B*cos_psi**2))))

cdef double c_get_vector_modulus(double *V, int len_arr):
	cdef double tot=0.0
	cdef int i
	for i in range(len_arr):
		tot+=V[i]*V[i]
	return sqrt(tot)

cdef void c_B_func(double r,double theta,double Bp, double *B_sph):
	B_sph[0]=Bp*(cos(theta))/r**3
	B_sph[1]=Bp*(0.5*sin(theta))/r**3
	B_sph[2]=0.0
	return	

cdef double c_get_cyclotron_freq(double r,double theta,float Bp):
	cdef double B
	cdef double B_vect[3]
	c_B_func(r,theta,Bp, B_vect)
	B	=c_get_vector_modulus(B_vect,3)	
	return 2.8*B #in MHz


cdef double delta_nu_p_delta_theta(double nu_p,double r,double theta,double phi,double n_p0,float r_max,float rho_fact): ##### To introduce non-ideal density distribution, already cythonized
	cdef double alpha, dn_dt
	alpha	=9.0*10.**(-3)
	dn_dt	=prop_tool_density.delta_n_delta_theta(n_p0,r,theta,phi,r_max,rho_fact) 
	return (alpha**2./(2.*nu_p))*dn_dt

cdef double delta_nu_p_delta_phi(double nu_p,double r,double theta,double phi,double n_p0,float r_max,float rho_fact): ##### To introduce non-ideal density distribution, already cythonized
	cdef double alpha, dn_dphi
	alpha	=9.0*10.**(-3)
	dn_dphi	=prop_tool_density.delta_n_delta_phi(n_p0,r,theta,phi,r_max,rho_fact) 
	return (alpha**2./(2.*nu_p))*dn_dphi

cdef double delta_nu_p_delta_theta_dash(double nu_p,double r,double theta,double phi): ##### To introduce non-ideal density distribution, already cythonized
	return 0.0

cdef double delta_nu_p_delta_phi_dash(double nu_p,double r,double theta,double phi): ##### To introduce non-ideal density distribution, already cythonized
	return 0.0

cdef get_cyclotron_freq(double r,double theta,float Bp):#alternate function written, but this is needed
	cdef double B
	cdef np.ndarray[np.double_t,ndim=1] B_vect
	B_vect	=B_func(r,theta,Bp)
	B	=get_vector_modulus(B_vect)	
	return 2.8*B #in MHz

cdef double delta_dot_kB_delta_theta(double r,double theta,double theta_dash,float Bp=1.0):  #already cythonized
	return (-sin(theta)+0.5*r*theta_dash*cos(theta))*(Bp/r**3)

cdef double delta_dot_kB_delta_theta_dash(double r,double theta,double theta_dash,float Bp=1.0):  #already cythonized
	return 0.5*r*sin(theta)*(Bp/r**3)

cdef double delta_k_delta_theta(double k,double r,double theta,double theta_dash,double phi_dash):  #already cythonized
	return (r**2*sin(theta)*cos(theta)*phi_dash**2)/k

cdef double delta_k_delta_theta_dash(double k,double r,double theta,double theta_dash,double phi_dash):  #already cythonized
	return (r**2*theta_dash)/k

cdef double delta_k_delta_phi_dash(double k,double r,double theta,double theta_dash,double phi_dash): #already cythonized
	return (r**2*sin(theta)**2*phi_dash)/k

cpdef get_cos_ang_bet_prop_vect_B_vect(double r,double theta,double phi,double theta_dash,double phi_dash):#alternate function written, but this is needed
	cdef np.ndarray[np.double_t, ndim=1] k_vect_volar, B_vect_polar
	cdef double k,B
	k_vect_polar	=np.array([1.,r*theta_dash,r*sin(theta)*phi_dash])
	B_vect_polar	=B_func(r,theta,1.0)
	k,B		=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	return np.dot(k_vect_polar,B_vect_polar)/(k*B)

cdef double delta_B_delta_theta(double r,double theta,float Bp):  #already cythonized
	return -(3/2.0)*(Bp/r**3)*(sin(theta)*cos(theta))/(sqrt(1+3*cos(theta)**2))


cdef double G_func(double r,double theta,double theta_dash,double phi_dash):  #already cythonized
	return sqrt(1+r**2*theta_dash**2+r**2*sin(theta)**2*phi_dash**2)


cdef double delta_G_delta_theta(double G,double r,double theta,double phi,double theta_dash,double phi_dash):  #already cythonized
	return (r**2*phi_dash**2*sin(2*theta))/(2*G)

cdef double delta_G_delta_theta_dash(double G,double r,double theta,double phi,double theta_dash,double phi_dash):  #already cythonized
	return (r**2*theta_dash)/G

cdef double delta_G_delta_phi_dash(double G,double r,double theta,double phi,double theta_dash,double phi_dash):  #already cythonized
	return (r**2*sin(theta)**2*phi_dash)/G

cdef double delta_nu_B_delta_theta(double r,double theta,double nu_B):  #already cythonized
	return (-3*nu_B*sin(theta)*cos(theta))/(1+3*cos(theta)**2)

cdef double delta_sigma_delta_theta(float nu,double nu_p,double nu_B,double cos_psi,double dnuB_dt,double dcos2psi_dt,double dnu_p_dt):  #already cythonized
	cdef double factor, factor1
	factor		=0.5*nu/abs(nu**2-nu_p**2)
	factor1		=0.5*nu*nu_B*(1-cos_psi**2)
	if nu>nu_p:
		return factor*(dnuB_dt*(1-cos_psi**2)-nu_B*dcos2psi_dt)+factor1*(2*nu_p*dnu_p_dt)/(nu**2-nu_p**2)**2
	else:
		return factor*(dnuB_dt*(1-cos_psi**2)-nu_B*dcos2psi_dt)-factor1*(2*nu_p*dnu_p_dt)/(nu**2-nu_p**2)**2

cdef double delta_sigma_delta_theta_dash(float nu,double nu_p,double nu_B,double cos_psi,double dnuB_dt_dash,double dcos2psi_dt_dash,double dnu_p_dt_dash):  #already cythonized
	return delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dt_dash,dcos2psi_dt_dash,dnu_p_dt_dash)
	
cdef double delta_sigma_delta_phi(float nu,double nu_p,double nu_B,double cos_psi,double dnuB_dphi,double dcos2psi_dphi,double dnu_p_dphi):  #already cythonized
	return delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dphi,dcos2psi_dphi,dnu_p_dphi)

cdef double delta_sigma_delta_phi_dash(float nu,double nu_p,double nu_B,double cos_psi,double dnuB_dphi_dash,double dcos2psi_dphi_dash,double dnu_p_dphi_dash):  #already cythonized
	return delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dphi_dash,dcos2psi_dphi_dash,dnu_p_dphi_dash)	

cdef double delta_tau_delta_theta(double sigma,double cos_psi,double dsigma_dt,double dcos2psi_dt,double nu_p,float nu): #already cythonized
	cdef double den
	den		=2*sqrt(sigma**2+cos_psi**2)
	if nu>nu_p:
		return -dsigma_dt-(2*sigma*dsigma_dt+dcos2psi_dt)/den
	else:
		return dsigma_dt+(sigma*dsigma_dt+dcos2psi_dt)/den

cdef double delta_tau_delta_theta_dash(double sigma,double cos_psi,double dsigma_dt_dash,double dcos2psi_dt_dash,double nu_p,float nu): #already cythonized
	return delta_tau_delta_theta(sigma,cos_psi,dsigma_dt_dash,dcos2psi_dt_dash,nu_p,nu)

cdef double delta_tau_delta_phi(double sigma,double cos_psi,double dsigma_dphi,double dcos2psi_dphi,double nu_p,float nu): #already cythonized
	return delta_tau_delta_theta(sigma,cos_psi,dsigma_dphi,dcos2psi_dphi,nu_p,nu)

cdef double delta_tau_delta_phi_dash(double sigma,double cos_psi,double dsigma_dphi_dash,double dcos2psi_dphi_dash,double nu_p,float nu): #already cythonized
	return delta_tau_delta_theta(sigma,cos_psi,dsigma_dphi_dash,dcos2psi_dphi_dash,nu_p,nu)	

cdef double delta_mu_delta_theta(double mu,double cos_psi,double nu_p,double nu_B,double tau,double dcos2psi_dt,double dnu_p_dt,double dnuB_dt,double dtau_dt,float nu,int mode): #already cythonized
	cdef double factor, factor1
	if mode==0:
		factor	=-nu_p**2/(2*mu*nu*(tau*nu-nu_B*cos_psi**2)**2)
		factor1	=-tau/(2*mu*nu*(tau*nu-nu_B*cos_psi**2))
		return factor*(-nu_B*cos_psi**2*dtau_dt+tau*nu_B*dcos2psi_dt+tau*cos_psi**2*dnuB_dt)+factor1*2*nu_p*dnu_p_dt
	else:
		factor	=nu_p**2/(2*mu*nu*(nu+tau*nu_B)**2)
		factor1	=-1/(2*mu*nu*(nu+tau*nu_B))
		return factor*(tau*dnuB_dt+nu_B*dtau_dt)+factor1*2*nu_p*dnu_p_dt

cdef double delta_mu_delta_phi(double mu,double cos_psi,double nu_p,double nu_B,double tau,double dcos2psi_dphi,double dnu_p_dphi,double dnuB_dphi,double dtau_dphi,float nu,int mode):#already cythonized
	return 	delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi,dnu_p_dphi,dnuB_dphi,dtau_dphi,nu,mode)

cdef double delta_mu_delta_theta_dash(double mu,double cos_psi,double nu_p,double nu_B,double tau,double dcos2psi_dt_dash,double dnu_p_dt_dash,double dnuB_dt_dash,double dtau_dt_dash,float nu,int mode):
	return delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dt_dash,dnu_p_dt_dash,dnuB_dt_dash,dtau_dt_dash,nu,mode) #already cythonized

cdef double delta_mu_delta_phi_dash(double mu,double cos_psi,double nu_p,double nu_B,double tau,double dcos2psi_dphi_dash,double dnu_p_dphi_dash,double dnuB_dphi_dash,double dtau_dphi_dash,float nu,int mode):
	return delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi_dash,dnu_p_dphi_dash,dnuB_dphi_dash,dtau_dphi_dash,nu,mode) #already cythonized

cdef double delta_F_delta_theta(double mu,double G,double cos_psi,double nu_p,double nu_B,double sigma,double tau,float nu,int mode,double r,double theta,double phi,double theta_dash,double phi_dash,double n_p0,float r_max,float rho_fact): #already_cythonized
	cdef double dnuB_dt, dcos2psi_dt, dnu_p_dt, dsigma_dt, dtau_dt, dmu_dtheta, dG_dtheta
	dnuB_dt		=delta_nu_B_delta_theta(r,theta,nu_B)
	dcos2psi_dt	=c_delta_cos_sq_psi_delta_theta(cos_psi,r,theta,theta_dash,phi_dash)
	dnu_p_dt	=delta_nu_p_delta_theta(nu_p,r,theta,phi,n_p0,r_max,rho_fact)
	dsigma_dt	=delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dt,dcos2psi_dt,dnu_p_dt)
	dtau_dt		=delta_tau_delta_theta(sigma,cos_psi,dsigma_dt,dcos2psi_dt,nu_p,nu)
	dmu_dtheta	=delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dt,dnu_p_dt,dnuB_dt,dtau_dt,nu,mode)
	dG_dtheta	=delta_G_delta_theta(G,r,theta,phi,theta_dash,phi_dash)
	return mu*dG_dtheta+G*dmu_dtheta


cdef double delta_F_delta_phi(double mu,double G,double cos_psi,double nu_p,double nu_B,double sigma,double tau,float nu,int mode,double r,double theta,double phi,double theta_dash,double phi_dash,double n_p0,float r_max,float rho_fact): #already_cythonized
	cdef double dnuB_dphi, dcos2psi_dphi, dnu_p_dphi, dsigma_dphi, dtau_dphi, dmu_dphi	
	dnuB_dphi		=0.0 #HARD-CODED
	dcos2psi_dphi		=c_delta_cos_sq_psi_delta_phi(cos_psi,r,theta,theta_dash,phi_dash)
	dnu_p_dphi		=delta_nu_p_delta_phi(nu_p,r,theta,phi,n_p0,r_max,rho_fact)
	dsigma_dphi		=delta_sigma_delta_phi(nu,nu_p,nu_B,cos_psi,dnuB_dphi,dcos2psi_dphi,dnu_p_dphi)
	dtau_dphi		=delta_tau_delta_phi(sigma,cos_psi,dsigma_dphi,dcos2psi_dphi,nu_p,nu)
	dmu_dphi		=delta_mu_delta_phi(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi,dnu_p_dphi,dnuB_dphi,dtau_dphi,nu,mode)
	return G*dmu_dphi

cdef double delta_F_delta_theta_dash(double mu,double G,double cos_psi,double nu_p,double nu_B,double sigma,double tau,float nu,int mode,double r,double theta,double phi,double theta_dash,double phi_dash):
	cdef double dnuB_dt_dash, dcos2psi_dt_dash, dnu_p_dt_dash, dsigma_dt_dash, dtau_dt_dash, dmu_dtheta_dash, dG_dtheta_dash  #already_cythonized
	dnuB_dt_dash	=0.0 #HARD-CODED
	dcos2psi_dt_dash=c_delta_cos_sq_psi_delta_theta_dash(cos_psi,r,theta,theta_dash,phi_dash)
	dnu_p_dt_dash	=delta_nu_p_delta_theta_dash(nu_p,r,theta,phi)
	dsigma_dt_dash	=delta_sigma_delta_theta_dash(nu,nu_p,nu_B,cos_psi,dnuB_dt_dash,dcos2psi_dt_dash,dnu_p_dt_dash)
	dtau_dt_dash	=delta_tau_delta_theta_dash(sigma,cos_psi,dsigma_dt_dash,dcos2psi_dt_dash,nu_p,nu)
	dmu_dtheta_dash	=delta_mu_delta_theta_dash(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dt_dash,dnu_p_dt_dash,dnuB_dt_dash,dtau_dt_dash,nu,mode)
	dG_dtheta_dash	=delta_G_delta_theta_dash(G,r,theta,phi,theta_dash,phi_dash)
	return mu*dG_dtheta_dash+G*dmu_dtheta_dash

cdef double delta_F_delta_phi_dash(double mu,double G,double cos_psi,double nu_p,double nu_B,double sigma,double tau,float nu,int mode,double r,double theta,double phi,double theta_dash,double phi_dash):
	cdef double dnuB_dphi_dash, dcos2psi_dphi_dash, dnu_p_dphi_dash, dsigma_dphi_dash, dtau_dphi_dash, dmu_dphi_dash, dG_dphi_dash  #already_cythonized
	dnuB_dphi_dash		=0.0 #HARD-CODED
	dcos2psi_dphi_dash	=c_delta_cos_sq_psi_delta_phi_dash(cos_psi,r,theta,theta_dash,phi_dash)
	dnu_p_dphi_dash		=delta_nu_p_delta_phi_dash(nu_p,r,theta,phi)
	dsigma_dphi_dash	=delta_sigma_delta_phi_dash(nu,nu_p,nu_B,cos_psi,dnuB_dphi_dash,dcos2psi_dphi_dash,dnu_p_dphi_dash)
	dtau_dphi_dash		=delta_tau_delta_phi_dash(sigma,cos_psi,dsigma_dphi_dash,dcos2psi_dphi_dash,nu_p,nu)
	dmu_dphi_dash=delta_mu_delta_phi_dash(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi_dash,dnu_p_dphi_dash,dnuB_dphi_dash,dtau_dphi_dash, nu,mode)
	dG_dphi_dash	=delta_G_delta_phi_dash(G,r,theta,phi,theta_dash,phi_dash)
	return mu*dG_dphi_dash+G*dmu_dphi_dash

cdef minimize_func(np.ndarray[np.double_t, ndim=1] theta_phi_dash,double r,double theta,double phi,np.ndarray[np.double_t, ndim=1] Y,double n_p0,float Bp,float nu,int mode,float r_max,double rho_fact):
	cdef double theta_dash, phi_dash, cos_psi, nu_p,nu_B, sigma, tau, mu, G, dF_dtheta_dash, dF_dphi_dash 
	theta_dash,phi_dash	=theta_phi_dash[0],theta_phi_dash[1]
	cos_psi			=get_cos_ang_bet_prop_vect_B_vect(r,theta,phi,theta_dash,phi_dash)
	nu_p			=get_plasma_freq(n_p0,r,theta,phi,r_max,rho_fact)
	nu_B			=get_cyclotron_freq(r,theta,Bp)
	sigma			=get_sigma(nu,nu_p,nu_B,cos_psi)
	tau			=get_tau(sigma,cos_psi,nu_p,nu)
	mu			=get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode)
	G			=G_func(r,theta,theta_dash,phi_dash)
	dF_dtheta_dash		=delta_F_delta_theta_dash(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash)
	dF_dphi_dash		=delta_F_delta_phi_dash(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash)
	return Y-np.array([dF_dtheta_dash,dF_dphi_dash])


###################
#ccccccccccccccccccccccccccccccccccccc


cdef double calc_dot_product(double *v1, double *v2, len_arr):
	cdef double tot=0.0
	cdef int i
	for i in range(len_arr):
		tot+=v1[i]*v2[i]
	return tot

cdef double c_delta_cos_sq_psi_delta_phi(double cos_psi,double r,double theta,double theta_dash,double phi_dash):
	cdef float Bp
	cdef double k, B, dot_kB, d_dot_kB_dphi, dk_dphi, dB_dphi
	cdef double k_vect_polar[3]
	cdef double B_vect_polar[3]
	Bp			=1.0
	k_vect_polar[:]		=[1,r*theta_dash,r*sin(theta)*phi_dash]
	c_B_func(r,theta,Bp,B_vect_polar)
	k			=c_get_vector_modulus(k_vect_polar,3)
	B			=c_get_vector_modulus(B_vect_polar,3)
	dot_kB			=calc_dot_product(k_vect_polar,B_vect_polar,3)
	d_dot_kB_dphi		=0.0
	dk_dphi			=0.0
	dB_dphi			=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dphi/dot_kB)-(dk_dphi/k)-(dB_dphi/B))

cdef double c_delta_cos_sq_psi_delta_theta(double cos_psi,double r,double theta,double theta_dash,double phi_dash):
	cdef double k, B, dot_kB, d_dot_kB_dt, dk_dt, dB_dt
	cdef float Bp
	cdef double k_vect_polar[3]
	cdef double B_vect_polar[3]
	Bp			=1.0
	k_vect_polar[:]		=[1,r*theta_dash,r*sin(theta)*phi_dash]
	c_B_func(r,theta,Bp,B_vect_polar)
	k			=c_get_vector_modulus(k_vect_polar,3)
	B			=c_get_vector_modulus(B_vect_polar,3)
	dot_kB			=calc_dot_product(k_vect_polar,B_vect_polar,3)
	d_dot_kB_dt		=delta_dot_kB_delta_theta(r,theta,theta_dash,Bp)
	dk_dt			=delta_k_delta_theta(k,r,theta,theta_dash,phi_dash)
	dB_dt			=delta_B_delta_theta(r,theta,Bp)
	return 2.*cos_psi**2*((d_dot_kB_dt/dot_kB)-(dk_dt/k)-(dB_dt/B))

cdef double c_delta_cos_sq_psi_delta_phi_dash(double cos_psi,double r,double theta,double theta_dash,double phi_dash):
	cdef double k, B, dot_kB, d_dot_kB_dphi_dash, dk_dphi_dash, dB_dphi_dash
	cdef float Bp
	cdef double k_vect_polar[3]
	cdef double B_vect_polar[3]
	Bp			=1.0
	k_vect_polar[:]		=[1,r*theta_dash,r*sin(theta)*phi_dash]
	c_B_func(r,theta,Bp,B_vect_polar)
	k			=c_get_vector_modulus(k_vect_polar,3)
	B			=c_get_vector_modulus(B_vect_polar,3)
	dot_kB			=calc_dot_product(k_vect_polar,B_vect_polar,3)
	d_dot_kB_dphi_dash	=0.0
	dk_dphi_dash		=delta_k_delta_phi_dash(k,r,theta,theta_dash,phi_dash)
	dB_dphi_dash		=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dphi_dash/dot_kB)-(dk_dphi_dash/k)-(dB_dphi_dash/B))

cdef double c_delta_cos_sq_psi_delta_theta_dash(double cos_psi,double r,double theta,double theta_dash,double phi_dash):
	cdef float Bp
	cdef double k, B, dot_kB, d_dot_kB_dt_dash, dk_dt_dash, dB_dt_dash
	cdef double k_vect_polar[3]
	cdef double B_vect_polar[3]
	Bp			=1.0
	k_vect_polar[:]		=[1,r*theta_dash,r*sin(theta)*phi_dash]
	c_B_func(r,theta,Bp,B_vect_polar)
	k			=c_get_vector_modulus(k_vect_polar,3)
	B			=c_get_vector_modulus(B_vect_polar,3)
	dot_kB			=calc_dot_product(k_vect_polar,B_vect_polar,3)
	d_dot_kB_dt_dash	=delta_dot_kB_delta_theta_dash(r,theta,theta_dash,Bp)
	dk_dt_dash		=delta_k_delta_theta_dash(k,r,theta,theta_dash,phi_dash)
	dB_dt_dash		=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dt_dash/dot_kB)-(dk_dt_dash/k)-(dB_dt_dash/B))

cdef void c_convert_spherical_to_cartesian_coord(double *V,double r, double theta, double phi, double *V_cart): 
	cdef double V_r, V_theta, V_phi
	V_r	=V[0]
	V_theta	=V[1]
	V_phi	=V[2]
	V_cart[0]	=V_r*sin(theta)*cos(phi)+V_theta*cos(theta)*cos(phi)-V_phi*sin(phi)
	V_cart[1]	=V_r*sin(theta)*sin(phi)+V_theta*cos(theta)*sin(phi)+V_phi*cos(phi)
	V_cart[2]	=V_r*cos(theta)-V_theta*sin(theta)
	return 

cdef void c_convert_cartesian_to_spherical_coord(double *V,double r,double theta,double phi, double *V_sph):
	cdef double V_x, V_y, V_z
	V_x,V_y,V_z		=V[0],V[1],V[2]
	V_sph[0]		=V_x*sin(theta)*cos(phi)+V_y*sin(theta)*sin(phi)+V_z*cos(theta)
	V_sph[1]		=V_x*cos(theta)*cos(phi)+V_y*cos(theta)*sin(phi)-V_z*sin(theta)
	V_sph[2]		=-V_x*sin(phi)+V_y*cos(phi)
	return 

cdef double c_get_cos_ang_bet_prop_vect_B_vect(double r,double theta,double phi,double theta_dash,double phi_dash):
	cdef double k_vect_polar[3]
	cdef double B_vect_polar[3]
	cdef double k,B
	k_vect_polar[:]	=[1.,r*theta_dash,r*sin(theta)*phi_dash]
	c_B_func(r,theta,1.0, B_vect_polar)
	k		=c_get_vector_modulus(k_vect_polar,3)
	B		=c_get_vector_modulus(B_vect_polar,3)
	return calc_dot_product(k_vect_polar,B_vect_polar,3)/(k*B) 

#ccccccccccccccccccccccccccccccccccccc
cpdef ray_path(double r0,double theta0,double phi0,double theta_dash0,double phi_dash0,float r_max,double n_p0,float Bp,float nu,int mode,float rho_fact, int len_r=10**5):
	#print(r0,theta0,phi0,theta_dash0,phi_dash0,r_max,n_p0,Bp,nu,mode,len_r)
	cdef double dr, z0, cos_psi0, nu_p0, nu_B0, sigma0, tau0, G0, mu0, rho, drho_dr, drho_dtheta, drho_dphi, l_r, cos_psi, nu_p, nu_B, sigma, tau, G
	cdef double delta_theta, delta_phi,l_theta,l_phi,dr_1,dr_0
	cdef np.ndarray[np.double_t, ndim=1] ini_guess,x0
	cdef np.ndarray[np.double_t, ndim=1] Y_arr
	cdef np.ndarray[np.double_t, ndim=1] my_r,my_theta,my_phi,my_mu,my_theta_dash,my_phi_dash
	#cdef bint cont
	cdef int i
	#cdef list r,theta,phi,mu,Y1,Y2, theta_dash,phi_dash, dy1_dr,dy2_dr
	cdef double k_ini[3]
	cdef double k_ini_cart[3]
	cdef double k_last[3]
	cdef double k_last_cart[3]
	cdef double *r, *theta, *phi, *mu, *Y1, *Y2, *theta_dash, *phi_dash, *dy1_dr, *dy2_dr
	cdef double rho_tol	=1e-11

	z0		=r0*cos(theta0)
	k_ini[:]	=[1,r0*theta_dash0,r0*sin(theta0)*phi_dash0]
	c_convert_spherical_to_cartesian_coord(k_ini,r0,theta0,phi0,k_ini_cart)

	r		=<double *>PyMem_Malloc(len_r*sizeof(double))
	theta		=<double *>PyMem_Malloc(len_r*sizeof(double))
	phi		=<double *>PyMem_Malloc(len_r*sizeof(double))
	mu		=<double *>PyMem_Malloc(len_r*sizeof(double))
	Y1		=<double *>PyMem_Malloc(len_r*sizeof(double))
	Y2		=<double *>PyMem_Malloc(len_r*sizeof(double))
	theta_dash	=<double *>PyMem_Malloc(len_r*sizeof(double))
	phi_dash	=<double *>PyMem_Malloc(len_r*sizeof(double))
	dy1_dr		=<double *>PyMem_Malloc(len_r*sizeof(double))
	dy2_dr		=<double *>PyMem_Malloc(len_r*sizeof(double))

	r[0]		=r0
	theta[0]	=theta0
	phi[0]		=phi0
	theta_dash[0]	=theta_dash0
	phi_dash[0]	=phi_dash0	

	cos_psi0	=c_get_cos_ang_bet_prop_vect_B_vect(r0,theta0,phi0,theta_dash0,phi_dash0)
	nu_p0		=get_plasma_freq(n_p0,r0,theta0,phi0,r_max,rho_fact)
	nu_B0		=c_get_cyclotron_freq(r0,theta0,Bp)
	sigma0		=get_sigma(nu,nu_p0,nu_B0,cos_psi0)
	tau0		=get_tau(sigma0,cos_psi0,nu_p0,nu)
	G0		=G_func(r0,theta0,theta_dash0,phi_dash0) 
	mu0		=get_refractive_index(nu_p0,nu_B0,tau0,cos_psi0,nu,mode)

	mu[0]		=mu0
	Y1[0]		=delta_F_delta_theta_dash(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0)
	Y2[0]		=delta_F_delta_phi_dash(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0)
	dy1_dr[0]	=delta_F_delta_theta(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0,n_p0,r_max,rho_fact)
	dy2_dr[0]	=delta_F_delta_phi(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0,n_p0,r_max,rho_fact)
	#print('Entering loop')
	for i in range(len_r-1):
		rho		=prop_tool_density.density_func(n_p0,r[i],theta[i],phi[i],r_max,rho_fact)
		drho_dr		=prop_tool_density.delta_n_delta_r(n_p0,r[i],theta[i],phi[i],r_max,rho_fact)
		drho_dtheta	=prop_tool_density.delta_n_delta_theta(n_p0,r[i],theta[i],phi[i],r_max,rho_fact)
		drho_dphi	=prop_tool_density.delta_n_delta_phi(n_p0,r[i],theta[i],phi[i],r_max,rho_fact)
		l_r		=abs(rho/drho_dr)
		dr_0		=l_r
		#print('\n',r[i],theta[i],phi[i],rho,l_r,i)
		
		if abs(drho_dtheta)>rho_tol:
			delta_theta	=rho/drho_dtheta
			l_theta	=abs(r[i]*delta_theta)
			dr_0	=min(l_r,l_theta)
		if abs(drho_dphi)>rho_tol:
			delta_phi	=rho/drho_dphi	
			l_phi	=abs(r[i]*np.sin(theta[i])*delta_phi)
			dr_0	=min(dr_0,l_phi)
		dr_1	=dr_0/1000.
		dr	=dr_1*(np.sqrt(r[i]**2+z0**2-2*r[i]*z0*np.cos(theta[i]))/(r[i]-z0*np.cos(theta[i])+r[i]*z0*np.sin(theta[i])*theta_dash[i]))


		#print(dr_0,dr_1,dr)		
		r[i+1]		=r[i]+dr
		theta[i+1]	=theta[i]+theta_dash[i]*dr
		phi[i+1]	=phi[i]+phi_dash[i]*dr
		Y1[i+1]		=Y1[i]+dy1_dr[i]*dr
		Y2[i+1]		=Y2[i]+dy2_dr[i]*dr
		ini_guess	=np.array([theta_dash[i],phi_dash[i]])
		Y_arr		=np.array([Y1[i+1],Y2[i+1]])
		ans		=optimize.root(minimize_func,x0=ini_guess,args=(r[i+1],theta[i+1],phi[i+1],Y_arr,n_p0,Bp,nu,mode,r_max,rho_fact))
		
		if ans['success']==True:
			theta_dash[i+1]	=ans['x'][0]
			phi_dash[i+1]	=ans['x'][1]
		else:
			return False,0

		cos_psi		=c_get_cos_ang_bet_prop_vect_B_vect(r[i+1],theta[i+1],phi[i+1],theta_dash[i+1],phi_dash[i+1])
		nu_p		=get_plasma_freq(n_p0,r[i+1],theta[i+1],phi[i+1],r_max,rho_fact)
		nu_B		=c_get_cyclotron_freq(r[i+1],theta[i+1],Bp)
		sigma		=get_sigma(nu,nu_p,nu_B,cos_psi)
		tau		=get_tau(sigma,cos_psi,nu_p,nu)
		mu[i+1]		=get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode)

		G		=G_func(r[i+1],theta[i+1],theta_dash[i+1],phi_dash[i+1])
		dy1_dr[i+1]	=delta_F_delta_theta(mu[i+1],G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r[i+1],theta[i+1],phi[i+1],theta_dash[i+1],phi_dash[i+1],n_p0,r_max,rho_fact)
		dy2_dr[i+1]	=delta_F_delta_phi(mu[i+1],G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r[i+1],theta[i+1],phi[i+1],theta_dash[i+1],phi_dash[i+1],n_p0,r_max,rho_fact)

		if (r[i+1]/(r_max*sin(theta[i+1])**2))>1:
			k_last[:]	=[1,r[i+1]*theta_dash[i+1],r[i+1]*sin(theta[i+1])*phi_dash[i+1]]
			c_convert_spherical_to_cartesian_coord(k_last,r[i+1],theta[i+1],phi[i+1],k_last_cart)
			break

	if (r[i+1]/(r_max*sin(theta[i+1])**2))<1:
		print('\n',r[i+1]/(r_max*sin(theta[i+1])**2,'The array length is insufficient to sample to inner magnetosphere. Increase len_r\n'))
		

		raise KeyboardInterrupt

	my_r,my_theta,my_phi,my_mu	=np.zeros(i+1),np.zeros(i+1),np.zeros(i+1),np.zeros(i+1)
	my_theta_dash,my_phi_dash	=np.zeros(i+1),np.zeros(i+1)

	for i in range(i+1):
		my_r[i]		=r[i]
		my_theta[i]	=theta[i]
		my_phi[i]	=phi[i]
		my_mu[i]	=mu[i]
		my_theta_dash[i]=theta_dash[i]
		my_phi_dash[i]	=phi_dash[i]
	
	PyMem_Free(r)
	PyMem_Free(theta)
	PyMem_Free(phi)
	PyMem_Free(theta_dash)
	PyMem_Free(phi_dash)
	PyMem_Free(mu)
	PyMem_Free(Y1)
	PyMem_Free(Y2)
	PyMem_Free(dy1_dr)
	PyMem_Free(dy2_dr)
	return True,np.array([my_r,my_theta,my_phi,my_theta_dash,my_phi_dash,my_mu])





