import h5py
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt

global interp_rho,interp_dn_dr,interp_dn_dtheta,interp_dn_dphi

filedir	='/home/dbarnali/postdoc/propagation_effects/input_output_for_density_grid/'

filename		='smoothed_rho_RRM_magnetic_frame_hd133880_bigger.h5'
f1			=h5py.File(filedir+filename,'r') 
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

'''
max_r	=24.0
mu_s	=np.sqrt(1-(1.0/max_r))

r   =np.linspace(1,24,500)
theta=np.linspace(np.arccos(mu_s),np.pi-np.arccos(mu_s),500)
phi =np.linspace(0,2*np.pi,100) #@@@@@@@@@ CAUTION: Phi is in radian


my_points   =[]
my_posx,my_posy,my_posz      =[],[],[]
for i in range(len(phi)):
    for j in range(len(r)):
        for k in range(len(theta)):
	    print(i,j,k)
            L   =r[j]/np.sin(theta[k])**2
            my_points.append(np.array([phi[i],r[j],theta[k]]))
            if L>max(r):
                my_posx.append(i)
                my_posy.append(j)
                my_posz.append(k)
            
                
new_rho=interp_rho(np.array(my_points)).reshape((len(phi),len(r),len(theta)))
new_rho[(my_posx,my_posy,my_posz)]=np.nan

plt.figure()
R,T	=np.meshgrid(r,theta)
XX,ZZ	=R*np.sin(T),R*np.cos(T)
plt.plot(max(r)*np.sin(theta)**2*np.sin(theta),max(r)*np.sin(theta)**2*np.cos(theta))
plt.contourf(XX,ZZ,new_rho[8,:,:].T,aspect='auto')
plt.colorbar()
plt.xlim(0,24)
plt.ylim(-12,12)
plt.show()
'''
