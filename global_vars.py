import h5py
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt

global interp_rho,interp_dn_dr,interp_dn_dtheta,interp_dn_dphi

filename		='smoothed_rho_RRM_magnetic_frame_hd133880_bigger.h5'   #density grid in the magnetic frame of reference
f1			=h5py.File(filename,'r') 
rho			=np.array(f1['rho']) #this is relative density
phi_arr,r_arr,theta_arr	=np.array(f1['phi_arr']),np.array(f1['r_arr']),np.array(f1['theta_arr'])
dn_dphi,dn_dr,dn_dtheta	=np.array(f1['drho_dphi']),np.array(f1['drho_dr']),np.array(f1['drho_dtheta'])
f1.close()

print('start interpolation')
print('r range',min(r_arr),max(r_arr))
print('theta range',min(theta_arr),max(theta_arr))
print('phi range',min(phi_arr),max(phi_arr))
###CAREFUL, Interpolation functions use extrapolation as well

interp_rho		=RegularGridInterpolator((phi_arr, r_arr, theta_arr), rho, bounds_error=True,fill_value=None)
interp_dn_dr		=RegularGridInterpolator((phi_arr, r_arr, theta_arr), dn_dr, bounds_error=True,fill_value=None)
interp_dn_dtheta    	=RegularGridInterpolator((phi_arr, r_arr, theta_arr), dn_dtheta, bounds_error=True,fill_value=None)
interp_dn_dphi		=RegularGridInterpolator((phi_arr, r_arr, theta_arr), dn_dphi, bounds_error=True,fill_value=None)
print('end interpolation')


