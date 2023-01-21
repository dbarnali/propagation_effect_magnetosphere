#This code finds out the initial condition to find out the ray path inside the inner magnetosphere, given a frequency nu, harmonic number
#harm_no and polar field strength Bp, assuming that the ray is originated along a field line given by r=Lsin^2theta
#The initial conditions are the starting values of the variables inside the inner magnetosphere 

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import odeint
from scipy.special import comb
import os
from scipy.interpolate import RegularGridInterpolator
import propagation_effect_tools_solve_integration_3D_grid as prop_tool_solve_int
from global_vars import *


def get_B_from_nu(nu,harm_no): #nu in MHz, B in gauss
	return nu/(2.8*harm_no)

def func1(theta,R_A,r01,theta01):
	return R_A*np.sin(theta)**2*np.cos(theta)-r01*np.cos(theta01)	

def func2(theta,L,Bp,B):	#nu_nuB=nu_B/nu_Bp=B/Bp	
	nu_nuB		=B/Bp
	return L**3*nu_nuB-np.sqrt(1-(3/4.0)*np.sin(theta)**2)/np.sin(theta)**6

def func3(phi,r0,theta0,r01,theta01,phi01,kx,ky):
	a=ky*(r0*np.sin(theta0)*np.cos(phi)-r01*np.sin(theta01)*np.cos(phi01))
	b=kx*(r0*np.sin(theta0)*np.sin(phi)-r01*np.sin(theta01)*np.sin(phi01))
	#print a+b
	return abs(a+b)

def get_sin_phi0(r0,theta0,r01,theta01,phi01,kx,ky):
	alpha	=r0*ky*np.sin(theta0)
	beta	=r0*kx*np.sin(theta0)
	gamma	=r01*ky*np.sin(theta01)*np.cos(phi01)-r01*kx*np.sin(theta01)*np.sin(phi01)
	temp	=alpha*np.sqrt(alpha**2+beta**2-gamma**2)
	den	=alpha**2+beta**2
	x1,x2	=((-gamma*beta)+temp)/den,((-gamma*beta)-temp)/den
	return x1,x2

	
def B_func(r,theta,Bp):
	Br,Bt=(np.cos(theta))/r**3,(0.5*np.sin(theta))/r**3
	return np.array([Br*Bp, Bt*Bp,0])

def convert_spherical_to_cartesian_coord(V,r,theta,phi): #here V is an array with components are the r,theta,phi components of vector V
	V_r,V_theta,V_phi	=V[0],V[1],V[2]
	V_x	=V_r*np.sin(theta)*np.cos(phi)+V_theta*np.cos(theta)*np.cos(phi)-V_phi*np.sin(phi)
	V_y	=V_r*np.sin(theta)*np.sin(phi)+V_theta*np.cos(theta)*np.sin(phi)+V_phi*np.cos(phi)
	V_z	=V_r*np.cos(theta)-V_theta*np.sin(theta)
	return np.array([V_x,V_y,V_z])

def convert_cartesian_to_spherical_coord(V,r,theta,phi):#here V is an array with components are the x,y,z components of vector V
	V_x,V_y,V_z	=V[0],V[1],V[2]
	V_r	=V_x*np.sin(theta)*np.cos(phi)+V_y*np.sin(theta)*np.sin(phi)+V_z*np.cos(theta)
	V_theta	=V_x*np.cos(theta)*np.cos(phi)+V_y*np.cos(theta)*np.sin(phi)-V_z*np.sin(theta)
	V_phi	=-V_x*np.sin(phi)+V_y*np.cos(phi)
	return np.array([V_r,V_theta,V_phi])

def get_vector_modulus(V):
	return np.sqrt(np.sum(V**2))

def get_normal_to_IM_boundary(theta,phi):
	n_r	=1
	n_theta	=-2/np.tan(theta)
	n_phi	=0.0
	return np.array([n_r,n_theta,n_phi])

def get_refraction_angle(mu1,mu2,inc_ang):
	if (mu1*np.sin(inc_ang))/mu2>1 or np.isnan(mu2)==True or np.isnan(inc_ang)==True :
		return False,0
	else:
		return True, np.arcsin((mu1*np.sin(inc_ang))/mu2)

def get_k_vector_inside_IM_polar(r,theta,phi,theta_dash,phi_dash): #in spherical polar coordinates
	return np.array([1,r*theta_dash,r*np.sin(theta)*phi_dash])

def func4(theta_phi_dash,r,theta,phi,k_01_polar,normal_vect_polar,inc_ang,theta_r):
	theta_dash,phi_dash	=theta_phi_dash[0],theta_phi_dash[1]
	k_0_polar		=get_k_vector_inside_IM_polar(r,theta,phi,theta_dash,phi_dash)
	k_01,k_0		=get_vector_modulus(k_01_polar),get_vector_modulus(k_0_polar)
	normal_vect		=get_vector_modulus(normal_vect_polar)
	v1			=np.dot(k_01_polar,k_0_polar)-k_01*k_0*np.cos(theta_r-inc_ang)
	v2			=np.dot(normal_vect_polar,k_0_polar)+normal_vect*k_0*np.cos(theta_r)
	return np.array([v1,v2])

def get_theta_dash0_phi_dash0(r,theta,phi,k_01_polar,normal_vect_polar,inc_ang,theta_r,tol=1e-6):
	k_r,k_theta,k_phi	=k_01_polar[0],k_01_polar[1],k_01_polar[2]
	n_r,n_theta		=normal_vect_polar[0],normal_vect_polar[1]
	k			=get_vector_modulus(k_01_polar)
	n			=get_vector_modulus(normal_vect_polar)
	A,B			=-n_r/(n*np.cos(theta_r)),-(r*n_theta)/(n*np.cos(theta_r))
	den			=r*k_phi*np.sin(theta)
	C,D			=(B*k*np.cos(theta_r-inc_ang)-r*k_theta)/den,(A*k*np.cos(theta_r-inc_ang)-k_r)/den	
	denom			=r*(-n_r*k_phi+C*np.sin(theta)*(n_r*k_theta-n_theta*k_r))
	numer			=-n_theta*k_phi-r*D*np.sin(theta)*(n_r*k_theta-n_theta*k_r)
	theta_dash0		=numer/denom
	phi_dash0		=C*theta_dash0+D
	return theta_dash0,phi_dash0

	
def check_soln_validity(k_01,r0,theta0,phi0,r01,theta01,phi01,tol=1e-10):
	kx,ky	=k_01[0],k_01[1]
	X1	=ky*(r0*np.sin(theta0)*np.cos(phi0)-r01*np.sin(theta01)*np.cos(phi01))
	X2	=kx*(r0*np.sin(theta0)*np.sin(phi0)-r01*np.sin(theta01)*np.sin(phi01))
	if abs(X1-X2)<tol:
		return True
	else:
		return False

def find_mu_in(mu,inc_ang,r0,theta0,phi0,nu,nu_p0,nu_B0,k_01_polar,normal_vect_polar,R_A,mode):
	theta_r	=np.arcsin(np.sin(inc_ang)/mu)
	theta_dash0,phi_dash0=get_theta_dash0_phi_dash0(r0,theta0,phi0,k_01_polar,normal_vect_polar,inc_ang,theta_r)
	cos_psi0=prop_tool_solve_int.get_cos_ang_bet_prop_vect_B_vect(r0,theta0,phi0,theta_dash0,phi_dash0)
	sigma0	=prop_tool_solve_int.get_sigma(nu,nu_p0,nu_B0,cos_psi0)
	tau0	=prop_tool_solve_int.get_tau(sigma0,cos_psi0,nu_p0,nu)
	new_mu	=prop_tool_solve_int.get_refractive_index(nu_p0,nu_B0,tau0,cos_psi0,nu,mode)
	return np.abs(mu-new_mu)

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

def get_initial_condition_array(n_p0,R_A,nu_B0,nu,mode,r0,theta0,phi0,k_01,cos_psi,rho_fact,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr):
	normal_vect_polar	=get_normal_to_IM_boundary(theta0,phi0)
	normal_vect_cart	=convert_spherical_to_cartesian_coord(normal_vect_polar,r0,theta0,phi0)
	inc_ang	=np.pi-np.arccos(np.dot(k_01,normal_vect_cart)/(get_vector_modulus(normal_vect_cart)*get_vector_modulus(k_01)))
	mu_1	=1.0


	####### NORTH POLE #######
	x1,y1,z1	=r0*np.sin(theta0)*np.cos(phi0),r0*np.sin(theta0)*np.sin(phi0),r0*np.cos(theta0)
	pos1		=find_pos_for_interp_func_arr(x1,y1,z1,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr)
	#print(pos1,len(interp_rho5_arr))
	interp_rho1	=interp_rho5_arr[pos1]	
	rho0	=prop_tool_solve_int.density_func(r0,theta0,phi0,n_p0,interp_rho1,rho_fact)	
	nu_p0	=prop_tool_solve_int.get_plasma_freq(rho0)
	sigma0	=prop_tool_solve_int.get_sigma(nu,nu_p0,nu_B0,cos_psi)
	tau0	=prop_tool_solve_int.get_tau(sigma0,cos_psi,nu_p0,nu)

	mu_2n	=prop_tool_solve_int.get_refractive_index(nu_p0,nu_B0,tau0,cos_psi,nu,mode)
	n_is_true,n_theta_r	=get_refraction_angle(mu_1,mu_2n,inc_ang)
	if n_is_true==True:
		k_01_polar	=convert_cartesian_to_spherical_coord(k_01,r0,theta0,phi0)
		ans	=optimize.root(find_mu_in,x0=mu_2n,args=(inc_ang,r0,theta0,phi0,nu,nu_p0,nu_B0,k_01_polar,normal_vect_polar,R_A,mode))
		if ans['success']==True:
			mu_2	=ans['x'][0]	
			n_theta_r	=np.arcsin(np.sin(inc_ang)/mu_2)
			theta_dash0,phi_dash0=get_theta_dash0_phi_dash0(r0,theta0,phi0,k_01_polar,normal_vect_polar,inc_ang,n_theta_r)
			n_ini_array	=np.array([r0,theta0,phi0,theta_dash0,phi_dash0])
			k_IM		=np.array([1,r0*theta_dash0,r0*np.sin(theta0)*phi_dash0])
			
		else:
			n_is_true=False
			n_ini_array=0.0
	else:
		n_ini_array=0.0


	######## SOUTH POLE ######
	x2,y2,z2	=r0*np.sin(theta0)*np.cos(phi0),r0*np.sin(theta0)*np.sin(phi0),-r0*np.cos(theta0)
	pos2		=find_pos_for_interp_func_arr(x2,y2,z2,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr)
	interp_rho2	=interp_rho5_arr[pos2]	

	rho0	=prop_tool_solve_int.density_func(r0,np.pi-theta0,phi0,n_p0,interp_rho2,rho_fact)	
	nu_p0	=prop_tool_solve_int.get_plasma_freq(rho0)
	sigma0	=prop_tool_solve_int.get_sigma(nu,nu_p0,nu_B0,cos_psi)
	tau0	=prop_tool_solve_int.get_tau(sigma0,cos_psi,nu_p0,nu)
	mu_2s	=prop_tool_solve_int.get_refractive_index(nu_p0,nu_B0,tau0,cos_psi,nu,mode)
	s_is_true,s_theta_r	=get_refraction_angle(mu_1,mu_2s,inc_ang)


	if s_is_true==True:
		normal_vect_polar	=get_normal_to_IM_boundary(np.pi-theta0,phi0)
		normal_vect_cart	=convert_spherical_to_cartesian_coord(normal_vect_polar,r0,theta0,phi0)
		
		k_01_polar	=convert_cartesian_to_spherical_coord(k_01,r0,np.pi-theta0,phi0)
		ans	=optimize.root(find_mu_in,x0=mu_2s,args=(inc_ang,r0,np.pi-theta0,phi0,nu,nu_p0,nu_B0,k_01_polar,normal_vect_polar,R_A,mode))
		if ans['success']==True:
			mu_2	=ans['x'][0]	
			s_theta_r	=np.arcsin(np.sin(inc_ang)/mu_2)
			theta_dash0,phi_dash0=get_theta_dash0_phi_dash0(r0,np.pi-theta0,phi0,k_01_polar,normal_vect_polar,inc_ang,s_theta_r)
			s_ini_array	=np.array([r0,np.pi-theta0,phi0,theta_dash0,phi_dash0])
			k_IM		=-np.array([1,r0*theta_dash0,r0*np.sin(theta0)*phi_dash0])
			#print np.dot(k_01_polar,k_IM)-get_vector_modulus(k_01_polar)*get_vector_modulus(k_IM)*np.cos(s_theta_r-inc_ang),'CHECK South'
			#print inc_ang*(180/np.pi),s_theta_r*(180/np.pi),'INC and REF ang'
			#print np.dot(normal_vect_polar,k_IM)+get_vector_modulus(normal_vect_polar)*get_vector_modulus(k_IM)*np.cos(s_theta_r),'CHECK South'
			#print np.dot(k_IM,np.cross(normal_vect_polar,k_01_polar)), theta_dash0,phi_dash0
			#print 1/(np.sin(theta0)+r0*np.cos(np.pi-theta0)*theta_dash0) ,'NEW'
		else:
			s_is_true=False
			s_ini_array=0.0


	else:
		s_ini_array=0.0

	return n_is_true,n_ini_array,s_is_true,s_ini_array

	

def get_initial_condition(nu,harm_no,Bp,L,R_A,phi01,n_p0,mode,rho_fact,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr):
	B		=get_B_from_nu(nu,harm_no) #B in gauss, nu in MHz
	h		=(Bp/B)**(1/3.0)
	ini_guess01	=np.arcsin(np.sqrt(h/L))
	ans		=optimize.root(func2,x0=ini_guess01,args=(L,Bp,B))

	if ans['success']==True:
		theta01	=ans['x'][0]	
	else:
		print(r'For nu='+str(nu)+' MHz, R_A='+str(R_A)+', L='+str(L)+',theta01 cannotbe found\nchange initial guess')
		raise KeyboardInterrupt

	r01		=L*np.sin(theta01)**2

	p	=np.array([1.0,0,-1.0,(r01*np.cos(theta01))/R_A])
	cos_theta0=np.roots(p)
	pos	=np.where(abs(cos_theta0)>1)[0]
	cos_theta0=np.delete(cos_theta0,pos)
	theta0_arr=np.arccos(cos_theta0)
	theta0	=min(theta0_arr)

	B_polar		=B_func(r01,theta01,Bp) #B vector in spherical polar coordinates
	B_cart		=convert_spherical_to_cartesian_coord(B_polar,r01,theta01,phi01)
	k_01		=np.array([B_cart[1],-B_cart[0],0.0]) #in cartesian coordinates
	
	r0		=R_A*np.sin(theta0)**2
	
	sin_phi0_1,sin_phi0_2=get_sin_phi0(r0,theta0,r01,theta01,phi01,k_01[0],k_01[1])
	B0_polar	=B_func(r0,theta0,Bp)	
	nu_B0		=prop_tool_solve_int.get_cyclotron_freq(r0,theta0,Bp)

	if (r0*np.sin(theta0)*sin_phi0_1-r01*np.sin(theta01)*np.sin(phi01))*k_01[1]>0:
		phi0_1,phi0_2	=np.arcsin(sin_phi0_1),np.pi-np.arcsin(sin_phi0_1)
		phi0=9999999
		if check_soln_validity(k_01,r0,theta0,phi0_1,r01,theta01,phi01,tol=1e-10)==True:
			phi0=phi0_1
		else:
			if check_soln_validity(k_01,r0,theta0,phi0_2,r01,theta01,phi01,tol=1e-10)==True:
				phi0=phi0_2
		if phi0==9999999:
			raise RuntimeError("Phi0 could not be solved. Please check\n")
		
	else:
		phi0_1,phi0_2	=np.arcsin(sin_phi0_2),np.pi-np.arcsin(sin_phi0_2)
		phi0=9999999
		if check_soln_validity(k_01,r0,theta0,phi0_1,r01,theta01,phi01,tol=1e-10)==True:
			phi0=phi0_1
		else:
			if check_soln_validity(k_01,r0,theta0,phi0_2,r01,theta01,phi01,tol=1e-10)==True:
				phi0=phi0_2
		if phi0==9999999:
			raise RuntimeError("Phi0 could not be solved. Please check\n")

	B0_cart		=convert_spherical_to_cartesian_coord(B0_polar,r0,theta0,phi0)
	cos_psi		=(np.dot(k_01,B0_cart)/(get_vector_modulus(B0_cart)*get_vector_modulus(k_01))) 
	#psi is the angle between propagation vector and B at the IM boundary just before entering

	phi0_1,phi0_2	=phi0,2*phi01-phi0
	n_is_true1,n_ini_arr1,s_is_true1,s_ini_arr1=get_initial_condition_array(n_p0,R_A,nu_B0,nu,mode,r0,theta0,phi0,k_01,cos_psi,rho_fact,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr)
	n_is_true2,n_ini_arr2,s_is_true2,s_ini_arr2=get_initial_condition_array(n_p0,R_A,nu_B0,nu,mode,r0,theta0,2*phi01-phi0,-k_01,cos_psi,rho_fact,min_x5_arr,min_y5_arr,min_z5_arr,max_x5_arr,max_y5_arr,max_z5_arr)
	#**IMP: we have used the azimuthal symmetry of the magnetic field by assuming that psi does not change 
	#for the oppositely travelling rays

	return [n_is_true1,n_is_true2,s_is_true1,s_is_true2],[n_ini_arr1,n_ini_arr2,s_ini_arr1,s_ini_arr2]


