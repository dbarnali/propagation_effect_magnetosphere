import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import odeint
from scipy.special import comb
from scipy.interpolate import RegularGridInterpolator
import os
from global_vars import *

def delta_x_delta_r(r,theta,phi):
	return np.sin(theta)*np.cos(phi)

def delta_y_delta_r(r,theta,phi):
	return np.sin(theta)*np.sin(phi)

def delta_z_delta_r(r,theta,phi):
	return np.cos(theta)

def delta_x_delta_theta(r,theta,phi):
	return r*np.cos(theta)*np.cos(phi)

def delta_y_delta_theta(r,theta,phi):
	return r*np.cos(theta)*np.sin(phi)

def delta_z_delta_theta(r,theta,phi):
	return -r*np.sin(theta)

def delta_x_delta_phi(r,theta,phi):
	return -r*np.sin(theta)*np.sin(phi)

def delta_y_delta_phi(r,theta,phi):
	return r*np.sin(theta)*np.cos(phi)

def density_func(r,theta,phi,n_p0,interp_rho,rho_fact):
	x,y,z=r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)
	rho	=n_p0*((1./r)+(rho_fact*interp_rho([x,y,z])[0]))
	return rho

def delta_n_delta_theta(r,theta,phi,n_p0,dn_dx,dn_dy,dn_dz,rho_fact): 
	#x,y,z			=r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)
	#dn_dx,dn_dy,dn_dz	=interp_dn_dx([x,y,z])[0],interp_dn_dy([x,y,z]),interp_dn_dz([x,y,z])
	dx_dtheta		=delta_x_delta_theta(r,theta,phi)
	dy_dtheta		=delta_y_delta_theta(r,theta,phi)
	dz_dtheta		=delta_z_delta_theta(r,theta,phi)
	dn_RRM			=dn_dx*dx_dtheta+dn_dy*dy_dtheta+dn_dz*dz_dtheta
	dn_dtheta		=n_p0*rho_fact*dn_RRM
	return dn_dtheta

def delta_n_delta_phi(r,theta,phi,n_p0,dn_dx,dn_dy,dn_dz,rho_fact): 
	#x,y,z			=r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)
	dx_dphi			=delta_x_delta_phi(r,theta,phi)
	dy_dphi			=delta_y_delta_phi(r,theta,phi)
	dz_dphi			=0.
	dn_RRM			=dn_dx*dx_dphi+dn_dy*dy_dphi+dn_dz*dz_dphi
	dn_dphi			=n_p0*rho_fact*dn_RRM
	return dn_dphi

def delta_n_delta_r(r,theta,phi,n_p0,dn_dx,dn_dy,dn_dz,rho_fact):
	#x,y,z			=r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)
	dx_dr			=delta_x_delta_r(r,theta,phi)
	dy_dr			=delta_y_delta_r(r,theta,phi)
	dz_dr			=delta_z_delta_r(r,theta,phi)
	dn_RRM			=dn_dx*dx_dr+dn_dy*dy_dr+dn_dz*dz_dr
	dn_dr			=n_p0*((-1./r**2)+(rho_fact*dn_RRM))
	return dn_dr


#############################################################
def get_plasma_freq(n_e):
	return 9000.0*10**(-6)*np.sqrt(n_e) # in MHz

def delta_nu_p_delta_theta(nu_p,dn_dtheta): ##### To introduce non-ideal density distribution
	alpha	=9000.0*10**(-6)
	return (alpha**2/(2*nu_p))*dn_dtheta

def delta_nu_p_delta_phi(nu_p,dn_dphi): ##### To introduce non-ideal density distribution
	alpha	=9000.0*10**(-6)
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

def find_pos_for_interp_func_arr(x,y,z,min_x_arr,min_y_arr,min_z_arr,max_x_arr,max_y_arr,max_z_arr):
	if x>=0:
		posx=np.where(max_x_arr>=x)[0]
	else:
		posx=np.where(min_x_arr<=x)[0]
	if y>=0:
		posy=np.where(max_y_arr>=y)[0]
	else:
		posy=np.where(min_y_arr<=y)[0]	
	if z>=0:
		posz=np.where(max_z_arr>=z)[0]
	else:
		posz=np.where(min_z_arr<=z)[0]
	'''
	pos1x	=np.where(min_x_arr<=x)[0]
	pos2x	=np.where(max_x_arr>=x)[0]
	posx	=np.intersect1d(pos1x,pos2x)
	pos1y	=np.where(min_y_arr<=y)[0]
	pos2y	=np.where(max_y_arr>=y)[0]
	posy	=np.intersect1d(pos1y,pos2y)
	
	pos1z	=np.where(min_z_arr<=z)[0]
	pos2z	=np.where(max_z_arr>=z)[0]
	posz	=np.intersect1d(pos1z,pos2z)
	'''
	posxy	=np.intersect1d(posx,posy)
	
	posxyz	=np.intersect1d(posxy,posz)[0]
	#print(posxyz)
	return posxyz


def minimize_func(theta_phi_dash,r,theta,phi,Y,Bp,nu,mode,n_p0,interp_rho,rho_fact):
	theta_dash,phi_dash	=theta_phi_dash[0],theta_phi_dash[1]
	cos_psi			=get_cos_ang_bet_prop_vect_B_vect(r,theta,phi,theta_dash,phi_dash)
	my_rho			=density_func(r,theta,phi,n_p0,interp_rho,rho_fact)	
	nu_p			=get_plasma_freq(my_rho)
	nu_B			=get_cyclotron_freq(r,theta,Bp)
	sigma			=get_sigma(nu,nu_p,nu_B,cos_psi)
	tau			=get_tau(sigma,cos_psi,nu_p,nu)
	mu			=get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode)
	G			=G_func(r,theta,theta_dash,phi_dash)
	dF_dtheta_dash		=delta_F_delta_theta_dash(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash)
	dF_dphi_dash		=delta_F_delta_phi_dash(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,theta_dash,phi_dash)
	return Y-np.array([dF_dtheta_dash,dF_dphi_dash])


def ray_path(r0,theta0,phi0,theta_dash0,phi_dash0,r_max,n_p0,Bp,nu,mode,rho_fact,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr,tol=1e-9):
	max_arr5			=min(max(max_x5_arr),max(max_y5_arr),max(max_z5_arr))

	z0	=r0*np.cos(theta0)
	r,theta,phi,mu,Y1,Y2	=[],[],[],[],[],[]
	theta_dash,phi_dash	=[],[]
	dy1_dr,dy2_dr		=[],[]

	r.append(r0)
	theta.append(theta0)
	phi.append(phi0)
	theta_dash.append(theta_dash0)
	phi_dash.append(phi_dash0)
	
	cos_psi0		=get_cos_ang_bet_prop_vect_B_vect(r0,theta0,phi0,theta_dash0,phi_dash0)
	x,y,z			=r0*np.sin(theta0)*np.cos(phi0),r0*np.sin(theta0)*np.sin(phi0),r0*np.cos(theta0)
	 
	if abs(x)<max_arr5 and abs(y)<max_arr5 and abs(z)<max_arr5:	
		i_pos			=find_pos_for_interp_func_arr(x,y,z,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr)
		interp_rho5		=interp_rho5_arr[i_pos]
		interp_dn_dx5		=interp_dn_dx5_arr[i_pos]
		interp_dn_dy5		=interp_dn_dy5_arr[i_pos]
		interp_dn_dz5		=interp_dn_dz5_arr[i_pos]

		rho0			=density_func(r0,theta0,phi0,n_p0,interp_rho5,rho_fact)
		dn_dx,dn_dy,dn_dz	=interp_dn_dx5([x,y,z])[0],interp_dn_dy5([x,y,z])[0],interp_dn_dz5([x,y,z])[0]
	else:
		rho0			=density_func(r0,theta0,phi0,n_p0,interp_rho20,rho_fact)
		dn_dx,dn_dy,dn_dz	=interp_dn_dx20([x,y,z])[0],interp_dn_dy20([x,y,z])[0],interp_dn_dz20([x,y,z])[0]	

	drho0_dr		=delta_n_delta_r(r0,theta0,phi0,n_p0,dn_dx,dn_dy,dn_dz,rho_fact)
	drho0_dtheta		=delta_n_delta_theta(r0,theta0,phi0,n_p0,dn_dx,dn_dy,dn_dz,rho_fact)
	drho0_dphi		=delta_n_delta_phi(r0,theta0,phi0,n_p0,dn_dx,dn_dy,dn_dz,rho_fact)
	
	nu_p0	=get_plasma_freq(rho0)
	nu_B0	=get_cyclotron_freq(r0,theta0,Bp)
	sigma0	=get_sigma(nu,nu_p0,nu_B0,cos_psi0)
	tau0	=get_tau(sigma0,cos_psi0,nu_p0,nu)
	G0	=G_func(r0,theta0,theta_dash0,phi_dash0) 
	mu0	=get_refractive_index(nu_p0,nu_B0,tau0,cos_psi0,nu,mode)

	mu.append(mu0)
	Y1.append(delta_F_delta_theta_dash(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0))
	Y2.append(delta_F_delta_phi_dash(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0))
	dy1_dr.append(delta_F_delta_theta(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0,drho0_dtheta))
	dy2_dr.append(delta_F_delta_phi(mu0,G0,cos_psi0,nu_p0,nu_B0,sigma0,tau0,nu,mode,r0,theta0,phi0,theta_dash0,phi_dash0,drho0_dphi))

	cont	=True
	i	=0
	while(cont==True):
		x,y,z			=r[i]*np.sin(theta[i])*np.cos(phi[i]),r[i]*np.sin(theta[i])*np.sin(phi[i]),r[i]*np.cos(theta[i])
		if abs(x)<max_arr5 and abs(y)<max_arr5 and abs(z)<max_arr5:
			i_pos			=find_pos_for_interp_func_arr(x,y,z,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr)
			#print(i_pos,len(interp_rho5_arr),'inside while loop',i,x,y,z)
			interp_rho5		=interp_rho5_arr[i_pos]
			#print(interp_rho5)
			interp_dn_dx5		=interp_dn_dx5_arr[i_pos]
			interp_dn_dy5		=interp_dn_dy5_arr[i_pos]
			interp_dn_dz5		=interp_dn_dz5_arr[i_pos]

			my_rho			=density_func(r[i],theta[i],phi[i],n_p0,interp_rho5,rho_fact)
			dn_dx,dn_dy,dn_dz	=interp_dn_dx5([x,y,z])[0],interp_dn_dy5([x,y,z])[0],interp_dn_dz5([x,y,z])[0]
		else:
			my_rho			=density_func(r[i],theta[i],phi[i],n_p0,interp_rho20,rho_fact)
			dn_dx,dn_dy,dn_dz	=interp_dn_dx20([x,y,z])[0],interp_dn_dy20([x,y,z])[0],interp_dn_dz20([x,y,z])[0]
	
		my_drho_dr		=delta_n_delta_r(r[i],theta[i],phi[i],n_p0,dn_dx,dn_dy,dn_dz,rho_fact)
		my_drho_dtheta		=delta_n_delta_theta(r[i],theta[i],phi[i],n_p0,dn_dx,dn_dy,dn_dz,rho_fact)
		my_drho_dphi		=delta_n_delta_phi(r[i],theta[i],phi[i],n_p0,dn_dx,dn_dy,dn_dz,rho_fact)
		
		length_scale	=[]
		l_r		=abs(my_rho/my_drho_dr)
		length_scale.append(l_r)
		if abs(my_drho_dtheta)>tol:
			delta_theta	=my_rho/my_drho_dtheta
			l_theta		=abs(r[i]*delta_theta)
			length_scale.append(l_theta)
		if abs(my_drho_dphi)>tol:
			delta_phi	=my_rho/my_drho_dphi
			l_phi		=abs(r[i]*np.sin(theta[i])*delta_phi)
			length_scale.append(l_phi)
		
		dr_1	=min(length_scale)/500.	#min(l_r,l_theta,l_phi)/200.
		dr	=dr_1*(np.sqrt(r[i]**2+z0**2-2*r[i]*z0*np.cos(theta[i]))/(r[i]-z0*np.cos(theta[i])+r[i]*z0*np.sin(theta[i])*theta_dash[i]))

		r.append(r[i]+dr)
		theta.append(theta[i]+theta_dash[i]*dr)
		phi.append(phi[i]+phi_dash[i]*dr)
		Y1.append(Y1[i]+dy1_dr[i]*dr)
		Y2.append(Y2[i]+dy2_dr[i]*dr)
		ini_guess	=np.array([theta_dash[i],phi_dash[i]])
		Y_arr		=np.array([Y1[i+1],Y2[i+1]])
		x,y,z		=r[i+1]*np.sin(theta[i+1])*np.cos(phi[i+1]),r[i+1]*np.sin(theta[i+1])*np.sin(phi[i+1]),r[i+1]*np.cos(theta[i+1])
		#print(x,y,z)
		if abs(x)<max_arr5 and abs(y)<max_arr5 and abs(z)<max_arr5:	
			i_pos			=find_pos_for_interp_func_arr(x,y,z,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr)
			
			interp_rho5		=interp_rho5_arr[i_pos]

			ans		=optimize.root(minimize_func,x0=ini_guess,args=(r[i+1],theta[i+1],phi[i+1],Y_arr,Bp,nu,mode,n_p0,interp_rho5,rho_fact))
		else:
			ans		=optimize.root(minimize_func,x0=ini_guess,args=(r[i+1],theta[i+1],phi[i+1],Y_arr,Bp,nu,mode,n_p0,interp_rho20,rho_fact))			
		if ans['success']==True:
			theta_dash.append(ans['x'][0])
			phi_dash.append(ans['x'][1])
			'''
			comp_num	=min(max_x5_arr)
			if x<0 and x>comp_num:
				print(x,y,z,i)
			if y<0 and y>comp_num:
				print(x,y,z,i)
			if z<0 and z>comp_num:
				print(x,y,z,i)
			'''
		else:
			
			#print(x,y,z,i,'\nFor nu='+str(nu)+' MHz, R_A='+str(r_max)+', np0='+str(n_p0)+' theta_dash & phi_dash cannot be found\nchange initial guess')
			#raise KeyboardInterrupt
			comp_num	=min(max_x5_arr)
			ch1	=x>0 or x<comp_num
			ch2	=y>0 or y<comp_num
			ch3	=z>0 or z<comp_num
			if ch1==True and ch2==True and ch3==True:
				print(i,x,y,z)
			'''
			if x>0 or x<comp_num:
				print(x,y,z,i)
			if y>0 or y<comp_num:
				print(x,y,z,i)
			if z>0 or z<comp_num:
				print(x,y,z,i)
			'''
			return False,0

		cos_psi		=get_cos_ang_bet_prop_vect_B_vect(r[i+1],theta[i+1],phi[i+1],theta_dash[i+1],phi_dash[i+1])
		
		if abs(x)<max_arr5 and abs(y)<max_arr5 and abs(z)<max_arr5:
			i_pos			=find_pos_for_interp_func_arr(x,y,z,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr)
			#print(i_pos,len(interp_rho5_arr),'further deeper inside while loop',i,x,y,z)
			interp_rho5		=interp_rho5_arr[i_pos]
			my_rho	=density_func(r[i+1],theta[i+1],phi[i+1],n_p0,interp_rho5,rho_fact)
		else:
			my_rho	=density_func(r[i+1],theta[i+1],phi[i+1],n_p0,interp_rho20,rho_fact)	
		nu_p		=get_plasma_freq(my_rho)
		nu_B		=get_cyclotron_freq(r[i+1],theta[i+1],Bp)
		sigma		=get_sigma(nu,nu_p,nu_B,cos_psi)
		tau		=get_tau(sigma,cos_psi,nu_p,nu)
		mu.append(get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode))

		G		=G_func(r[i+1],theta[i+1],theta_dash[i+1],phi_dash[i+1])
		dy1_dr.append(delta_F_delta_theta(mu[i+1],G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r[i+1],theta[i+1],phi[i+1],theta_dash[i+1],phi_dash[i+1],my_drho_dtheta))
		dy2_dr.append(delta_F_delta_phi(mu[i+1],G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r[i+1],theta[i+1],phi[i+1],theta_dash[i+1],phi_dash[i+1],my_drho_dphi))
	
		if (r[i+1]/(r_max*np.sin(theta[i+1])**2))<1:
			cont=True
			i+=1
		else:
			cont=False
	
	r,theta,phi,mu		=np.array(r),np.array(theta),np.array(phi),np.array(mu)
	theta_dash,phi_dash	=np.array(theta_dash),np.array(phi_dash)
	
	return True,np.array([r,theta,phi,theta_dash,phi_dash,mu])



