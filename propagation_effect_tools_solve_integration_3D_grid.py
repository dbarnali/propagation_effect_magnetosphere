import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy.special import comb
from scipy.interpolate import RegularGridInterpolator
import os
from global_vars import *
import propagation_effect_tools_density_func as prop_tool_density

theta_phi_dash0=np.zeros(2)
#############################################################
def get_plasma_freq(n_e):
	return 9.0*10.**(-3)*np.sqrt(n_e) # in MHz

def delta_nu_p_delta_theta(nu_p,dn_dtheta): ##### To introduce non-ideal density distribution
	alpha	=9.0*10.**(-3)
	return (alpha**2/(2*nu_p))*dn_dtheta

def delta_nu_p_delta_phi(nu_p,dn_dphi): ##### To introduce non-ideal density distribution
	alpha	=9.0*10.**(-3)
	return (alpha**2/(2*nu_p))*dn_dphi

def delta_nu_p_delta_theta_dash(nu_p,r,theta,phi): ##### To introduce non-ideal density distribution
	return 0.0

def delta_nu_p_delta_phi_dash(nu_p,r,theta,phi): ##### To introduce non-ideal density distribution
	return 0.0

def get_cyclotron_freq(r,theta,Bp):
	B_vect	=B_func(r,theta,Bp)
	B	=get_vector_modulus(B_vect)	
	return 2.8*B #in MHz

def delta_dot_kB_delta_theta(r,theta,theta_dash,Bp=1.0):
	return (-np.sin(theta)+0.5*r*theta_dash*np.cos(theta))*(Bp/r**3)

def delta_dot_kB_delta_theta_dash(r,theta,theta_dash,Bp=1.0):
	return 0.5*r*np.sin(theta)*(Bp/r**3)

def delta_k_delta_theta(k,r,theta,theta_dash,phi_dash):
	return (r**2*np.sin(theta)*np.cos(theta)*phi_dash**2)/k

def delta_k_delta_theta_dash(k,r,theta,theta_dash,phi_dash):
	return (r**2*theta_dash)/k

def delta_k_delta_phi_dash(k,r,theta,theta_dash,phi_dash):
	return (r**2*np.sin(theta)**2*phi_dash)/k

def get_cos_ang_bet_prop_vect_B_vect(r,theta,phi,theta_dash,phi_dash):
	if len(np.shape(r))>0:
		print('get_cos_ang_bet_prop_vect_B_vect')
	k_vect_polar	=np.array([1,r*theta_dash,r*np.sin(theta)*phi_dash])
	B_vect_polar	=B_func(r,theta,1.0)
	k,B		=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	return np.dot(k_vect_polar,B_vect_polar)/(k*B)

def delta_cos_sq_psi_delta_theta(cos_psi,r,theta,theta_dash,phi_dash):
	Bp			=1.0
	if len(np.shape(r))>0:
		print('delta_cos_sq_psi_delta_theta')
	k_vect_polar		=np.array([1,r*theta_dash,r*np.sin(theta)*phi_dash])
	B_vect_polar		=B_func(r,theta,Bp)
	k,B			=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	dot_kB			=np.dot(k_vect_polar,B_vect_polar)
	d_dot_kB_dt		=delta_dot_kB_delta_theta(r,theta,theta_dash,Bp)
	dk_dt			=delta_k_delta_theta(k,r,theta,theta_dash,phi_dash)
	dB_dt			=delta_B_delta_theta(r,theta,Bp)
	return 2*cos_psi**2*((d_dot_kB_dt/dot_kB)-(dk_dt/k)-(dB_dt/B))

def delta_cos_sq_psi_delta_phi(cos_psi,r,theta,theta_dash,phi_dash):
	Bp			=1.0
	if len(np.shape(r))>0:
		print('delta_cos_sq_psi_delta_phi')
	k_vect_polar		=np.array([1,r*theta_dash,r*np.sin(theta)*phi_dash])
	B_vect_polar		=B_func(r,theta,Bp)
	k,B			=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	dot_kB			=np.dot(k_vect_polar,B_vect_polar)
	d_dot_kB_dphi		=0.0
	dk_dphi			=0.0
	dB_dphi			=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dphi/dot_kB)-(dk_dphi/k)-(dB_dphi/B))

def delta_cos_sq_psi_delta_theta_dash(cos_psi,r,theta,theta_dash,phi_dash):
	Bp			=1.0
	if len(np.shape(r))>0:
		print('delta_cos_sq_psi_delta_theta_dash')
	k_vect_polar		=np.array([1,r*theta_dash,r*np.sin(theta)*phi_dash])
	B_vect_polar		=B_func(r,theta,Bp)
	k,B			=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	dot_kB			=np.dot(k_vect_polar,B_vect_polar)
	d_dot_kB_dt_dash	=delta_dot_kB_delta_theta_dash(r,theta,theta_dash,Bp)
	dk_dt_dash		=delta_k_delta_theta_dash(k,r,theta,theta_dash,phi_dash)
	dB_dt_dash		=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dt_dash/dot_kB)-(dk_dt_dash/k)-(dB_dt_dash/B))

def delta_cos_sq_psi_delta_phi_dash(cos_psi,r,theta,theta_dash,phi_dash):
	Bp			=1.0
	if len(np.shape(r))>0:
		print('delta_cos_sq_psi_delta_phi_dash')
	k_vect_polar		=np.array([1,r*theta_dash,r*np.sin(theta)*phi_dash])
	B_vect_polar		=B_func(r,theta,Bp)
	k,B			=get_vector_modulus(k_vect_polar),get_vector_modulus(B_vect_polar)
	dot_kB			=np.dot(k_vect_polar,B_vect_polar)
	d_dot_kB_dphi_dash	=0.0
	dk_dphi_dash		=delta_k_delta_phi_dash(k,r,theta,theta_dash,phi_dash)
	dB_dphi_dash		=0.0 #HARD-CODED
	return 2*cos_psi**2*((d_dot_kB_dphi_dash/dot_kB)-(dk_dphi_dash/k)-(dB_dphi_dash/B))

def B_func(r,theta,Bp):
	Br,Bt=(np.cos(theta))/r**3,(0.5*np.sin(theta))/r**3
	return np.array([Br*Bp, Bt*Bp,0])

def delta_B_delta_theta(r,theta,Bp):
	return -(3/2.0)*(Bp/r**3)*(np.sin(theta)*np.cos(theta))/(np.sqrt(1+3*np.cos(theta)**2))

def get_vector_modulus(V):
	return np.sqrt(np.sum(V**2))

def get_sigma(nu,nu_p,nu_B,cos_psi):
	factor		=0.5*nu/abs(nu**2-nu_p**2)
	return 	factor*nu_B*(1-cos_psi**2)

def get_tau(sigma,cos_psi,nu_p,nu):
	if nu>nu_p:
		return -(sigma+np.sqrt(sigma**2+cos_psi**2))
	else:
		return (sigma+np.sqrt(sigma**2+cos_psi**2))
	
def G_func(r,theta,theta_dash,phi_dash):
	return np.sqrt(1+r**2*theta_dash**2+r**2*np.sin(theta)**2*phi_dash**2)

def get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode):
	if mode==1:	# X-mode
		return np.sqrt(1-(nu_p**2/(nu*(nu+tau*nu_B))))
	else:		# O-mode
		return np.sqrt(1-((tau*nu_p**2)/(nu*(tau*nu-nu_B*cos_psi**2))))


def delta_G_delta_theta(G,r,theta,phi,theta_dash,phi_dash):
	return (r**2*phi_dash**2*np.sin(2*theta))/(2*G)

def delta_G_delta_theta_dash(G,r,theta,phi,theta_dash,phi_dash):
	return (r**2*theta_dash)/G

def delta_G_delta_phi_dash(G,r,theta,phi,theta_dash,phi_dash):
	return (r**2*np.sin(theta)**2*phi_dash)/G

def delta_nu_B_delta_theta(r,theta,nu_B):
	return (-3*nu_B*np.sin(theta)*np.cos(theta))/(1+3*np.cos(theta)**2)

def delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dt,dcos2psi_dt,dnu_p_dt):
	factor		=0.5*nu/abs(nu**2-nu_p**2)
	factor1		=0.5*nu*nu_B*(1-cos_psi**2)
	if nu>nu_p:
		return factor*(dnuB_dt*(1-cos_psi**2)-nu_B*dcos2psi_dt)+factor1*(2*nu_p*dnu_p_dt)/(nu**2-nu_p**2)**2
	else:
		return factor*(dnuB_dt*(1-cos_psi**2)-nu_B*dcos2psi_dt)-factor1*(2*nu_p*dnu_p_dt)/(nu**2-nu_p**2)**2

def delta_sigma_delta_theta_dash(nu,nu_p,nu_B,cos_psi,dnuB_dt_dash,dcos2psi_dt_dash,dnu_p_dt_dash):
	return delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dt_dash,dcos2psi_dt_dash,dnu_p_dt_dash)
	
def delta_sigma_delta_phi(nu,nu_p,nu_B,cos_psi,dnuB_dphi,dcos2psi_dphi,dnu_p_dphi):
	return delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dphi,dcos2psi_dphi,dnu_p_dphi)

def delta_sigma_delta_phi_dash(nu,nu_p,nu_B,cos_psi,dnuB_dphi_dash,dcos2psi_dphi_dash,dnu_p_dphi_dash):
	return delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dphi_dash,dcos2psi_dphi_dash,dnu_p_dphi_dash)	

def delta_tau_delta_theta(sigma,cos_psi,dsigma_dt,dcos2psi_dt,nu_p,nu):
	den		=2*np.sqrt(sigma**2+cos_psi**2)
	if nu>nu_p:
		return -dsigma_dt-(2*sigma*dsigma_dt+dcos2psi_dt)/den
	else:
		return dsigma_dt+(sigma*dsigma_dt+dcos2psi_dt)/den

def delta_tau_delta_theta_dash(sigma,cos_psi,dsigma_dt_dash,dcos2psi_dt_dash,nu_p,nu):
	return delta_tau_delta_theta(sigma,cos_psi,dsigma_dt_dash,dcos2psi_dt_dash,nu_p,nu)

def delta_tau_delta_phi(sigma,cos_psi,dsigma_dphi,dcos2psi_dphi,nu_p,nu):
	return delta_tau_delta_theta(sigma,cos_psi,dsigma_dphi,dcos2psi_dphi,nu_p,nu)

def delta_tau_delta_phi_dash(sigma,cos_psi,dsigma_dphi_dash,dcos2psi_dphi_dash,nu_p,nu):
	return delta_tau_delta_theta(sigma,cos_psi,dsigma_dphi_dash,dcos2psi_dphi_dash,nu_p,nu)	

def delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dt,dnu_p_dt,dnuB_dt,dtau_dt,nu,mode):
	if mode==0:
		factor	=-nu_p**2/(2*mu*nu*(tau*nu-nu_B*cos_psi**2)**2)
		factor1	=-tau/(2*mu*nu*(tau*nu-nu_B*cos_psi**2))
		return factor*(-nu_B*cos_psi**2*dtau_dt+tau*nu_B*dcos2psi_dt+tau*cos_psi**2*dnuB_dt)+factor1*2*nu_p*dnu_p_dt
	else:
		factor	=nu_p**2/(2*mu*nu*(nu+tau*nu_B)**2)
		factor1	=-1/(2*mu*nu*(nu+tau*nu_B))
		return factor*(tau*dnuB_dt+nu_B*dtau_dt)+factor1*2*nu_p*dnu_p_dt

def delta_mu_delta_phi(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi,dnu_p_dphi,dnuB_dphi,dtau_dphi,nu,mode):
	return 	delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi,dnu_p_dphi,dnuB_dphi,dtau_dphi,nu,mode)

def delta_mu_delta_theta_dash(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dt_dash,dnu_p_dt_dash,dnuB_dt_dash,dtau_dt_dash,nu,mode):
	return delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dt_dash,dnu_p_dt_dash,dnuB_dt_dash,dtau_dt_dash,nu,mode)

def delta_mu_delta_phi_dash(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi_dash,dnu_p_dphi_dash,dnuB_dphi_dash,dtau_dphi_dash,nu,mode):
	return delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi_dash,dnu_p_dphi_dash,dnuB_dphi_dash,dtau_dphi_dash,nu,mode)

def delta_F_delta_theta(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash,dn_dtheta):
	dnuB_dt		=delta_nu_B_delta_theta(r,theta,nu_B)
	dcos2psi_dt	=delta_cos_sq_psi_delta_theta(cos_psi,r,theta,theta_dash,phi_dash)
	dnu_p_dt	=delta_nu_p_delta_theta(nu_p,dn_dtheta)
	dsigma_dt	=delta_sigma_delta_theta(nu,nu_p,nu_B,cos_psi,dnuB_dt,dcos2psi_dt,dnu_p_dt)
	dtau_dt		=delta_tau_delta_theta(sigma,cos_psi,dsigma_dt,dcos2psi_dt,nu_p,nu)
	dmu_dtheta	=delta_mu_delta_theta(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dt,dnu_p_dt,dnuB_dt,dtau_dt,nu,mode)
	dG_dtheta	=delta_G_delta_theta(G,r,theta,phi,theta_dash,phi_dash)
	return mu*dG_dtheta+G*dmu_dtheta

def delta_F_delta_phi(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash,dn_dphi):
	dnuB_dphi		=0.0 #HARD-CODED
	dcos2psi_dphi		=delta_cos_sq_psi_delta_phi(cos_psi,r,theta,theta_dash,phi_dash)
	dnu_p_dphi		=delta_nu_p_delta_phi(nu_p,dn_dphi)
	dsigma_dphi		=delta_sigma_delta_phi(nu,nu_p,nu_B,cos_psi,dnuB_dphi,dcos2psi_dphi,dnu_p_dphi)
	dtau_dphi		=delta_tau_delta_phi(sigma,cos_psi,dsigma_dphi,dcos2psi_dphi,nu_p,nu)
	dmu_dphi		=delta_mu_delta_phi(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi,dnu_p_dphi,dnuB_dphi,dtau_dphi,nu,mode)        
	return G*dmu_dphi

def delta_F_delta_theta_dash(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash):
	dnuB_dt_dash	=0.0 #HARD-CODED
	dcos2psi_dt_dash=delta_cos_sq_psi_delta_theta_dash(cos_psi,r,theta,theta_dash,phi_dash)
	dnu_p_dt_dash	=delta_nu_p_delta_theta_dash(nu_p,r,theta,phi)
	dsigma_dt_dash	=delta_sigma_delta_theta_dash(nu,nu_p,nu_B,cos_psi,dnuB_dt_dash,dcos2psi_dt_dash,dnu_p_dt_dash)
	dtau_dt_dash	=delta_tau_delta_theta_dash(sigma,cos_psi,dsigma_dt_dash,dcos2psi_dt_dash,nu_p,nu)
	dmu_dtheta_dash	=delta_mu_delta_theta_dash(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dt_dash,dnu_p_dt_dash,dnuB_dt_dash,dtau_dt_dash,nu,mode)
	dG_dtheta_dash	=delta_G_delta_theta_dash(G,r,theta,phi,theta_dash,phi_dash)
	return mu*dG_dtheta_dash+G*dmu_dtheta_dash

def delta_F_delta_phi_dash(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash):
	dnuB_dphi_dash		=0.0 #HARD-CODED
	dcos2psi_dphi_dash	=delta_cos_sq_psi_delta_phi_dash(cos_psi,r,theta,theta_dash,phi_dash)
	dnu_p_dphi_dash		=delta_nu_p_delta_phi_dash(nu_p,r,theta,phi)
	dsigma_dphi_dash	=delta_sigma_delta_phi_dash(nu,nu_p,nu_B,cos_psi,dnuB_dphi_dash,dcos2psi_dphi_dash,dnu_p_dphi_dash)
	dtau_dphi_dash		=delta_tau_delta_phi_dash(sigma,cos_psi,dsigma_dphi_dash,dcos2psi_dphi_dash,nu_p,nu)
	
	dmu_dphi_dash=delta_mu_delta_phi_dash(mu,cos_psi,nu_p,nu_B,tau,dcos2psi_dphi_dash,dnu_p_dphi_dash,dnuB_dphi_dash,dtau_dphi_dash,nu,mode)
	dG_dphi_dash	=delta_G_delta_phi_dash(G,r,theta,phi,theta_dash,phi_dash)
	return mu*dG_dphi_dash+G*dmu_dphi_dash



def minimize_func(theta_phi_dash,r,theta,phi,Y,n_p0,Bp,nu,mode,r_max,rho_fact):
	theta_dash,phi_dash	=theta_phi_dash[0],theta_phi_dash[1]
	cos_psi			=get_cos_ang_bet_prop_vect_B_vect(r,theta,phi,theta_dash,phi_dash)
	my_rho			=prop_tool_density.density_func(n_p0,r,theta,phi,r_max,rho_fact)	
	nu_p			=get_plasma_freq(my_rho)
	nu_B			=get_cyclotron_freq(r,theta,Bp)
	sigma			=get_sigma(nu,nu_p,nu_B,cos_psi)
	tau			=get_tau(sigma,cos_psi,nu_p,nu)
	mu			=get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode)
	G			=G_func(r,theta,theta_dash,phi_dash)
	dF_dtheta_dash		=delta_F_delta_theta_dash(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash)
	dF_dphi_dash		=delta_F_delta_phi_dash(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash)
	return Y-np.array([dF_dtheta_dash,dF_dphi_dash])

def test_func(r,full_y_arr,n_p0, Bp, nu, mode, r_max, rho_fact):
	#print(theta_phi_dash0,r)
	theta,phi,Y1,Y2 =full_y_arr[0],full_y_arr[1],full_y_arr[2],full_y_arr[3]
	Y_arr		=np.array([Y1,Y2])
	ini_guess       =np.array([theta_phi_dash0[0],theta_phi_dash0[1]])
	ans		=optimize.root(minimize_func,x0=ini_guess,args=(r,theta,phi,Y_arr,n_p0,Bp,nu,mode,r_max,rho_fact))		
	if ans['success']==True:
		theta_phi_dash0[0]	=ans['x'][0]
		theta_phi_dash0[1]	=ans['x'][1]
		dtheta_dr  =theta_phi_dash0[0]
		dphi_dr  =theta_phi_dash0[1]
		cos_psi		=get_cos_ang_bet_prop_vect_B_vect(r,theta,phi,dtheta_dr,dphi_dr)
		my_rho		=prop_tool_density.density_func(n_p0,r,theta,phi,r_max,rho_fact)
		nu_p		=get_plasma_freq(my_rho)
		nu_B		=get_cyclotron_freq(r,theta,Bp)
		sigma		=get_sigma(nu,nu_p,nu_B,cos_psi)
		tau		=get_tau(sigma,cos_psi,nu_p,nu)
		mu		=get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode)
		G		=G_func(r,theta,dtheta_dr,dphi_dr)
		dn_dtheta       =prop_tool_density.delta_n_delta_theta(n_p0, r,  theta,  phi, r_max,  rho_fact)
		dn_dphi         =prop_tool_density.delta_n_delta_phi(n_p0, r,  theta,  phi, r_max,  rho_fact)
		dy1_dr	        =delta_F_delta_theta(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,dtheta_dr,dphi_dr,dn_dtheta)
		dy2_dr	        =delta_F_delta_phi(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,dtheta_dr,dphi_dr,dn_dphi)

		return np.array([dtheta_dr,dphi_dr,dy1_dr,dy2_dr])

	else:
		print('theta_dash and phi_dash could not be solved')
		raise KeyboardInterrupt

def convert_spherical_to_cartesian_coord(V,r,theta,phi): 
	V_r	=V[0]
	V_theta	=V[1]
	V_phi	=V[2]
	V_cart  =np.zeros(3)
	V_cart[0]	=V_r*np.sin(theta)*np.cos(phi)+V_theta*np.cos(theta)*np.cos(phi)-V_phi*np.sin(phi)
	V_cart[1]	=V_r*np.sin(theta)*np.sin(phi)+V_theta*np.cos(theta)*np.sin(phi)+V_phi*np.cos(phi)
	V_cart[2]	=V_r*np.cos(theta)-V_theta*np.sin(theta)
	return V_cart

def ray_path_using_scipy(r0, theta0, phi0, theta_dash0, phi_dash0, r_max, n_p0, Bp, nu, mode, rho_fact, len_r=10**3):
	#print(r0,theta0,phi0,theta_dash0,phi_dash0,r_max,n_p0,Bp,nu,mode,len_r)
	theta_phi_dash0[0]=theta_dash0
	theta_phi_dash0[1]=phi_dash0
	z0		=r0*np.cos(theta0)
	k_ini   	=np.array([1,r0*theta_dash0,r0*np.sin(theta0)*phi_dash0])
	k_ini_cart      =convert_spherical_to_cartesian_coord(k_ini,r0,theta0,phi0)

	cos_psi0	=get_cos_ang_bet_prop_vect_B_vect(r0,theta0,phi0,theta_dash0,phi_dash0)
	rho0            =prop_tool_density.density_func(n_p0,r0,theta0,phi0,r_max,rho_fact)
	nu_p0		=get_plasma_freq(rho0)
	nu_B0		=get_cyclotron_freq(r0,theta0,Bp)
	sigma0		=get_sigma(nu,nu_p0,nu_B0,cos_psi0)
	tau0		=get_tau(sigma0,cos_psi0,nu_p0,nu)
	G0		=G_func(r0,theta0,theta_dash0,phi_dash0) 
	mu0		=get_refractive_index(nu_p0,nu_B0,tau0,cos_psi0,nu,mode)

	Y10		=delta_F_delta_theta_dash(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0)
	Y20		=delta_F_delta_phi_dash(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0)

	full_y0_arr     =np.array([theta0,phi0,Y10,Y20])
	val             =ode(test_func).set_integrator('vode',method='bdf')
	val.set_initial_value(full_y0_arr,r0).set_f_params(n_p0, Bp,nu, mode,r_max, rho_fact)
	r               =np.linspace(r0,r_max,len_r)
	theta           =np.zeros(len_r)
	phi             =np.zeros(len_r)
	theta_dash      =np.zeros(len_r)
	phi_dash        =np.zeros(len_r)
	mu              =np.zeros(len_r)
	dt              =r[1]-r[0]
	theta[0],phi[0] =theta0,phi0
	theta_dash[0]   =theta_dash0
	phi_dash[0]     =phi_dash0
	mu[0]           =mu0
	count           =1
	while(val.successful()==True):
		#print('my r value is',r[count-1])
		ans =val.integrate(val.t+dt)
		theta[count]    =ans[0]
		phi[count]      =ans[1]
		Y_arr   =np.array([ans[2],ans[3]])
		#print(theta_phi_dash0)
		new_ans =optimize.root(minimize_func,x0=theta_phi_dash0,args=(r[count],theta[count],phi[count],Y_arr,n_p0,Bp,nu,mode,r_max,rho_fact))
		if new_ans['success']==True:
			theta_dash[count]   =new_ans['x'][0]
			phi_dash[count]     =new_ans['x'][1]
			cos_psi		    =get_cos_ang_bet_prop_vect_B_vect(r[count],theta[count],phi[count],theta_dash[count],phi_dash[count])
			rho                 =prop_tool_density.density_func(n_p0,r[count],theta[count],phi[count],r_max,rho_fact)
			nu_p		    =get_plasma_freq(rho)
			nu_B		    =get_cyclotron_freq(r[count],theta[count],Bp)
			sigma		    =get_sigma(nu,nu_p,nu_B,cos_psi)
			tau		    =get_tau(sigma,cos_psi,nu_p,nu)
			mu[count]	    =get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode)
			theta_phi_dash0[0]  =theta_dash[count]
			theta_phi_dash0[1]  =phi_dash[count]
			count               =count+1
		else:
			print('theta_dash and phi_dash could not be solved')
			raise KeyboardInterrupt

		if  (r[count-1]/(r_max*np.sin(theta[count-1])**2))>1 or count==len_r:
			#print('count=',count,'I have reached the edge')
			break

	if (r[count-1]/(r_max*np.sin(theta[count-1])**2))<1:
		print('\n',r[count-1]/(r_max*np.sin(theta[count-1])**2,'The array length is insufficient to sample to inner magnetosphere. Increase len_r\n'))
		raise KeyboardInterrupt
	if count<len_r:
		r           =r[:count]
		theta       =theta[:count]
		phi         =phi[:count]
		theta_dash  =theta_dash[:count]
		phi_dash    =phi_dash[:count]
		mu          =mu[:count]
    
	return True,np.array([r,theta,phi,theta_dash,phi_dash,mu])





