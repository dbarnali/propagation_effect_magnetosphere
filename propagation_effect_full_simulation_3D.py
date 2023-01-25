import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import odeint
from scipy.special import comb
import os
import sys
import propagation_effect_tools_solve_integration_3D_grid as prop_tool_solve_int
os.system('python3 setup_cypython.py build_ext --inplace')


import propagation_effect_tools_initial_condition_3D_complex_IM_cython as prop_tool_ini_cond
import propagation_effect_tools_density_func as prop_tool_density

####import propagation_effect_tools_solve_integration_3D_simple_IM as prop_tool_solve_int
####import propagation_effect_tools_find_initial_condition_3D_complex_IM as prop_tool_ini_cond

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
import matplotlib as mpl
import time
from pylab import text
import pickle
from multiprocessing import Pool
import time
import h5py
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
from global_vars import *
start_time = time.time()

mpl.rcParams['font.size'] = 15

def find_ray_path_given_ini_cond(
        pole,
        phi01,
        ini_is_true,
        ini_arr,
        mode,
        harm_no,
        nu,
        R_A,
        L,
        n_p0,
        Bp,
        rho_fact):
    #print(nu, ini_is_true)
    if ini_is_true == 1:
        r0, theta0, phi0, theta_dash0, phi_dash0 = ini_arr

        #print('Calling ray path')
        is_true, full_ray_path = prop_tool_solve_int.ray_path_using_scipy(
            r0, theta0, phi0, theta_dash0, phi_dash0, R_A, n_p0, Bp, nu, mode, rho_fact, 10**2)
        # print(is_true)
        if is_true:
            r, theta, phi, theta_dash, phi_dash, mu = full_ray_path
            pos = np.where((r / (R_A * np.sin(theta)**2)) <= 1.0)[0]
            r, theta, phi, dtheta_dr, dphi_dr, mu = r[pos], theta[
                pos], phi[pos], theta_dash[pos], phi_dash[pos], mu[pos]
            ray_path = np.array([r, theta, phi, dtheta_dr, dphi_dr, mu])
        else:
            print(
                '\nno ray path for phi01=', str(
                    phi01 * 180 / np.pi), ' from pole ' + pole)
            ray_path = 0
    else:
        print('\nno ray path for phi01=', str(
            phi01 * 180 / np.pi), ' from pole' + pole)
        is_true = False
        ray_path = 0.0
    return is_true, ray_path


def main_func(phi01, mode, harm_no, nu, R_A, L, n_p0, Bp, rho_fact):
    print(phi01, (L / R_A) - 1, nu)
    ini_is_true_arr, ini_arr = prop_tool_ini_cond.get_initial_condition(
        nu, harm_no, Bp, L, R_A, phi01, n_p0, mode, rho_fact)
    #print(ini_is_true_arr,'initial_condition')

    n_is_true1, n_ray_path1 = find_ray_path_given_ini_cond(
        'N', phi01, ini_is_true_arr[0], ini_arr[0], mode, harm_no, nu, R_A, L, n_p0, Bp, rho_fact)
    #print(phi01, 'I am back')
    n_is_true2, n_ray_path2 = find_ray_path_given_ini_cond(
        'N', phi01, ini_is_true_arr[1], ini_arr[1], mode, harm_no, nu, R_A, L, n_p0, Bp, rho_fact)
    #print(phi01, 'I am back2')
    s_is_true1, s_ray_path1 = find_ray_path_given_ini_cond(
        'S', phi01, ini_is_true_arr[2], ini_arr[2], mode, harm_no, nu, R_A, L, n_p0, Bp, rho_fact)
    #print(phi01, 'I am back3')
    s_is_true2, s_ray_path2 = find_ray_path_given_ini_cond(
        'S', phi01, ini_is_true_arr[3], ini_arr[3], mode, harm_no, nu, R_A, L, n_p0, Bp, rho_fact)
    #print(phi01, 'I am back4')
    return [n_is_true1, n_is_true2, s_is_true1, s_is_true2], [
        n_ray_path1, n_ray_path2, s_ray_path1, s_ray_path2]


def get_k_vect_outside_IM(
        r_exit,
        theta_exit,
        phi_exit,
        dtheta_dr_exit,
        dphi_dr_exit,
        mu_exit):
    normal_vect_polar = - \
        prop_tool_ini_cond.get_normal_to_IM_boundary(theta_exit, phi_exit)
    k_inside_IM_polar = prop_tool_ini_cond.get_k_vector_inside_IM_polar(
        r_exit, theta_exit, phi_exit, dtheta_dr_exit, dphi_dr_exit)
    n, k = prop_tool_ini_cond.get_vector_modulus(
        normal_vect_polar), prop_tool_ini_cond.get_vector_modulus(k_inside_IM_polar)
    inc_ang_exit = np.pi - \
        np.arccos(np.dot(k_inside_IM_polar, normal_vect_polar) / (k * n))
    mu_out = 1.0
    is_true, theta_r_out = prop_tool_ini_cond.get_refraction_angle(
        mu_exit, mu_out, inc_ang_exit)
    x1, x2 = prop_tool_ini_cond.get_theta_dash0_phi_dash0(
        r_exit, theta_exit, phi_exit, k_inside_IM_polar, normal_vect_polar, inc_ang_exit, theta_r_out, tol=1e-6)
    k_out_polar = np.array(
        [1.0, r_exit * x1, r_exit * np.sin(theta_exit) * x2])
    k_out_cart = prop_tool_ini_cond.convert_spherical_to_cartesian_coord(
        k_out_polar, r_exit, theta_exit, phi_exit)
    k_out = prop_tool_ini_cond.get_vector_modulus(k_out_cart)
    thetaz = np.arccos(k_out_cart[-1] / k_out)
    return k_out_cart, thetaz


def find_rot_phase(D, alpha, beta):
    phi_rot = (1 / (2.0 * np.pi)) * np.arccos((np.sin(D) - np.cos(beta)
                                               * np.cos(alpha)) / (np.sin(beta) * np.sin(alpha)))
    return phi_rot


def find_lag(nu, phi_rot_mean):  # here nu is an array
    lag = []
    nu1, nu2 = [], []
    for i in range(len(nu)):
        for j in range(i + 1, len(nu)):
            nu1.append(nu[i])
            nu2.append(nu[j])
            lag.append(phi_rot_mean[j] - phi_rot_mean[i])
    return np.array(nu1), np.array(nu2), np.array(lag)


def find_density(n_p0, r, theta, phi, r_max):
    ne = np.zeros(len(phi))
    for j in range(len(phi)):
        ne[j] = prop_tool_density.density_func(
            n_p0, r[j], theta[j], phi[j], r_max)
    return ne


def append_ray_path(is_true, ray_path, pole, n_p0, R_A):
    if is_true:
        r1, theta1, phi1, dtheta_dr1, dphi_dr1, mu1 = ray_path
        ne1 = find_density(n_p0, r1, theta1, phi1, R_A)
        k_out_cart, thetaz = get_k_vect_outside_IM(
            r1[-1], theta1[-1], phi1[-1], dtheta_dr1[-1], dphi_dr1[-1], mu1[-1])
        if pole == 1:
            n_ray_path.append(ray_path)
            n_ane.append(ne1)
            n_ak_out.append(k_out_cart)
        else:
            s_ray_path.append(ray_path)
            s_ane.append(ne1)
            s_ak_out.append(k_out_cart)


def get_LOS_in_B_frame_cart(alpha, beta, rot_phase):
    phi_rot = 2 * np.pi * rot_phase
    LOSx, LOSy = -np.sin(alpha) * np.sin(phi_rot), -np.sin(alpha) * \
        np.cos(beta) * np.cos(phi_rot) + np.cos(alpha) * np.sin(beta)
    LOSz = np.sin(alpha) * np.sin(beta) * np.cos(phi_rot) + \
        np.cos(alpha) * np.cos(beta)
    return np.array([LOSx, LOSy, LOSz])


def call_main_func(args):
    phiBs = args[0]
    mode = args[1]
    harm_no = args[2]
    nu = args[3]
    R_A = args[4]
    L = args[5]
    n_p0 = args[6]
    Bp = args[7]
    core_id = args[8]
    rho_fact = args[9]
    nu0 = (min(nu) + max(nu)) / 2

    n_ray_path, s_ray_path = [], []

    for j in range(len(L)):
        for k in range(len(nu)):
            for i in range(len(phiBs)):
                is_true, ray_path = main_func(
                    phiBs[i], mode, harm_no, nu[k], R_A, L[j], n_p0, Bp, rho_fact)
                #print('\n*********************', L[j], nu[k], phiBs[i],j,k,i)
                if is_true[0]:
                    n_ray_path.append(ray_path[0])
                if is_true[1]:
                    n_ray_path.append(ray_path[1])
                if is_true[2]:
                    s_ray_path.append(ray_path[2])
                if is_true[3]:
                    s_ray_path.append(ray_path[3])

    output_ray_paths = {}
    output_ray_paths['N_ray_path'] = n_ray_path
    output_ray_paths['S_ray_path'] = s_ray_path
    if mode == 1:
        filename = 'outputs_X_mode_' + \
            str(int(nu0)) + 'MHz_core_' + str(core_id) + '.p'
    if mode == 0:
        filename = 'outputs_O_mode_' + \
            str(int(nu0)) + 'MHz_core_' + str(core_id) + '.p'

    pickle.dump(output_ray_paths, open(filename, 'wb'))


star_name    ='hd133880'
#alpha,beta              =46.5*(np.pi/180.0),76.*(np.pi/180.0)	#@@@@@@@@@@@@@@CHECK
alpha,beta              =55.*(np.pi/180.0),78.*(np.pi/180.0)
#alpha,beta              =54.6*(np.pi/180.0),80.*(np.pi/180.0)	#@@@@
Bp = 9600.	#4000.0
n_p0 = 10**8
mode = 1.  # 0 means O-mode and 1 means X-mode
harm_no = 2.01  # harmonic number, must be >1 for X mode
R_A =15.0#20.
# l_R_A			=np.linspace(20.,30,5)
# l_R_A			=np.array([.1])
L = [18.0]#np.linspace(25., 30, 1)  # (1+l_R_A)*R_A
nu0 = 600.
dnu = 234 / 2.
nu = [nu0]#np.linspace(nu0 - dnu, nu0 + dnu, 1)
rho_fact =0.01
out_dir         ='/home/dbarnali/postdoc/propagation_effects/input_output_for_density_grid/'

if mode == 1:
    out_filename =out_dir+ 'cython_X_mode_' + str(int(nu0)) + 'MHz_density_RRM_'+star_name+'_like_rho_fact_'+str(rho_fact)+'.p'
if mode == 0:
    out_filename =out_dir+ 'cython_O_mode_' + str(int(nu0)) + 'MHz_density_RRM_'+star_name+'_like_rho_fact_'+str(rho_fact)+'.p'

# phiB	=np.array([1.8824224168936852])	#np.array([86.*(np.pi/180.)])
len_phib = 180
#phiB	=np.array([np.pi/2.])
phiB = np.linspace(0.001, 2 * np.pi - 0.001, len_phib)
num_cores = min(10, len_phib)
phib_per_core = len_phib // num_cores if len_phib % num_cores == 0 else len_phib // num_cores + 1

args_list = []
for i in range(num_cores):
    try:
        args_list.append([phiB[i * phib_per_core:(i + 1) * phib_per_core],
                         mode, harm_no, nu, R_A, L, n_p0, Bp, i, rho_fact])
    except IndexError:
        args_list.append([phiB[i * phib_per_core:len_phib],
                         mode, harm_no, nu, R_A, L, n_p0, Bp, i, rho_fact])


parallel_instance = Pool(num_cores)
success = parallel_instance.map(call_main_func, args_list)


print("--- %s seconds ---" % (time.time() - start_time))


n_ray_path, s_ray_path = [], []

for i in range(num_cores):
    core_id = i
    if mode == 1:
        filename = 'outputs_X_mode_' + \
            str(int(nu0)) + 'MHz_core_' + str(core_id) + '.p'
    if mode == 0:
        filename = 'outputs_O_mode_' + \
            str(int(nu0)) + 'MHz_core_' + str(core_id) + '.p'

    output = pickle.load(open(filename, 'rb'))
    n_ray = output['N_ray_path']
    s_ray = output['S_ray_path']
    for j in range(len(n_ray)):
        n_ray_path.append(n_ray[j])
    for j in range(len(s_ray)):
        s_ray_path.append(s_ray[j])

output_ray_paths = {}
output_ray_paths['N_ray_path'] = n_ray_path
output_ray_paths['S_ray_path'] = s_ray_path
'''
hf      =h5py.File(out_filename.split('.')[0]+'.h5','w')
hf.create_dataset('N_ray_path',data=n_ray_path)
hf.create_dataset('S_ray_path',data=s_ray_path)
hf.close()
'''
pickle.dump(output_ray_paths, open(out_filename, 'wb'))

os.system('rm outputs*core*.p')

###output = pickle.load(open(out_filename, 'rb'))
###n_ray_path= output['N_ray_path']
###s_ray_path= output['S_ray_path']

n_k_out_cart, s_k_out_cart = [], []
n_k_out_mod, s_k_out_mod = np.zeros(len(n_ray_path)), np.zeros(len(s_ray_path))

for i in range(len(n_ray_path)):
    r, theta, phi, dtheta_dr, dphi_dr, mu = n_ray_path[i]
    k_out_cart, thetaz = get_k_vect_outside_IM(
        r[-1], theta[-1], phi[-1], dtheta_dr[-1], dphi_dr[-1], mu[-1])
    n_k_out_mod[i] = prop_tool_ini_cond.get_vector_modulus(k_out_cart)
    n_k_out_cart.append(k_out_cart)


for i in range(len(s_ray_path)):
    r, theta, phi, dtheta_dr, dphi_dr, mu = s_ray_path[i]
    k_out_cart, thetaz = get_k_vect_outside_IM(
        r[-1], theta[-1], phi[-1], dtheta_dr[-1], dphi_dr[-1], mu[-1])
    s_k_out_mod[i] = prop_tool_ini_cond.get_vector_modulus(k_out_cart)
    s_k_out_cart.append(k_out_cart)

rot_phase = np.linspace(0.5, 1.5, 1000)
n_flux, s_flux = np.zeros(len(rot_phase)), np.zeros(len(rot_phase))
sigma_theta = 3 * (np.pi / 180.)
B_curve = np.zeros(len(rot_phase))
ax = plt.subplot(111)
for i in range(len(rot_phase)):
    LOS_cart = get_LOS_in_B_frame_cart(alpha, beta, rot_phase[i])
    LOS = prop_tool_ini_cond.get_vector_modulus(LOS_cart)
    B_curve[i] = LOS_cart[-1]

    for j in range(len(n_k_out_cart)):
        k_out_cart = n_k_out_cart[j]
        k_out = n_k_out_mod[j]
        my_theta = np.arccos(np.dot(LOS_cart, k_out_cart) / (LOS * k_out))
        n_flux[i] += np.exp(-my_theta**2 / (sigma_theta**2))
    for j in range(len(s_k_out_cart)):
        k_out_cart = s_k_out_cart[j]
        k_out = s_k_out_mod[j]
        my_theta = np.arccos(np.dot(LOS_cart, k_out_cart) / (LOS * k_out))
        s_flux[i] += np.exp(-my_theta**2 / (sigma_theta**2))

max_flux = max(max(n_flux), max(s_flux))
print(max(n_flux), max(s_flux))

plt.plot(rot_phase, n_flux / max_flux, 'r-',
         lw=1, label='lightcurve: north pole')
plt.plot(rot_phase, s_flux / max_flux, 'b-',
         lw=1, label='lightcurve: south pole')
plt.plot(rot_phase, B_curve, 'k--', lw=2, label=r'$B_\mathrm{LoS}$')
plt.xlabel('Rotational Phase')
plt.legend(loc='best')
plt.tight_layout()
# plt.xlim(0.2,0.8)
#text(0.5,0.8,'2 GHz, X-mode',ha='center', va='center', transform=ax.transAxes, fontsize=25)
plt.grid()
#plt.savefig("cython_density_func10_ne_1e9_fig_" + str(int(nu0)) + ".png")
# plt.close()
plt.show()
