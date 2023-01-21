import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import odeint
from scipy.special import comb
import os
import propagation_effect_tools_solve_integration_3D_complex_IM_cython as prop_tool_solve_int
import propagation_effect_tools_initial_condition_3D_complex_IM_cython as prop_tool_ini_cond
import propagation_effect_tools_density_func as prop_tool_density
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
import matplotlib as mpl
import time
from pylab import text
import pickle
from multiprocessing import Pool
import time
start_time = time.time()

mpl.rcParams['font.size'] = 15


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


def get_LOS_in_B_frame_cart(alpha, beta, rot_phase):
    phi_rot = 2 * np.pi * rot_phase
    LOSx, LOSy = -np.sin(alpha) * np.sin(phi_rot), -np.sin(alpha) * \
        np.cos(beta) * np.cos(phi_rot) + np.cos(alpha) * np.sin(beta)
    LOSz = np.sin(alpha) * np.sin(beta) * np.cos(phi_rot) + \
        np.cos(alpha) * np.cos(beta)
    return np.array([LOSx, LOSy, LOSz])


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


def func(theta01, r0, theta0, L):
    #print L * np.sin(theta01)**2 * np.cos(theta01) - r0 * np.cos(theta0)
    return L * np.sin(theta01)**2 * np.cos(theta01) - r0 * np.cos(theta0)


def find_phi01_given_phi0(r0, theta0, phi0, theta_dash0, phi_dash0, R_A, L):
    if theta0 > np.pi / 2.:
        theta0 = np.pi - theta0
    k_IM_polar = np.array(
        [1., r0 * theta_dash0, r0 * np.sin(theta0) * phi_dash0])
    k_IM_cart = prop_tool_ini_cond.convert_spherical_to_cartesian_coord(
        k_IM_polar, r0, theta0, phi0)
    k_IM = prop_tool_ini_cond.get_vector_modulus(k_IM_cart)

    p = np.array([1.0, 0, -1.0, (r0 * np.cos(theta0)) / L])

    cos_theta01 = np.roots(p)
    pos = np.where(abs(cos_theta01) > 1)[0]
    cos_theta01 = np.delete(cos_theta01, pos)
    theta01_arr = np.arccos(cos_theta01)
    theta01 = min(theta01_arr)
    r01 = L * np.sin(theta01)**2
    phi1 = np.arccos((r01 * np.sin(theta01)) / (r0 * np.sin(theta0)))
    phi01_1, phi01_2 = phi0 + phi1, phi0 - phi1

    if phi01_1 < 0:
        phi01_1 += 2 * np.pi
    if phi01_2 < 0:
        phi01_2 += 2 * np.pi

    k1_cart = np.array([r0 *
                        np.sin(theta0) *
                        np.cos(phi0) -
                        r01 *
                        np.sin(theta01) *
                        np.cos(phi01_1), r0 *
                        np.sin(theta0) *
                        np.sin(phi0) -
                        r01 *
                        np.sin(theta01) *
                        np.sin(phi01_1), 0])
    k2_cart = np.array([r0 *
                        np.sin(theta0) *
                        np.cos(phi0) -
                        r01 *
                        np.sin(theta01) *
                        np.cos(phi01_2), r0 *
                        np.sin(theta0) *
                        np.sin(phi0) -
                        r01 *
                        np.sin(theta01) *
                        np.sin(phi01_2), 0])
    k1, k2 = prop_tool_ini_cond.get_vector_modulus(
        k1_cart), prop_tool_ini_cond.get_vector_modulus(k2_cart)
    ang1, ang2 = np.arccos(np.dot(k_IM_cart, k1_cart) / (k_IM * k1)
                           ), np.arccos(np.dot(k_IM_cart, k2_cart) / (k_IM * k2))
    if ang1 < ang2:
        return phi01_1
    else:
        return phi01_2


def delta_n_delta_r(np0, r, theta, phi, r_max):
    theta0 = (np.pi / 2.0) * (1 + np.sin(phi))
    r0 = 2.5
    M = 5
    y = 1 / (1 + np.exp(2 * M * (r - r0)))
    z = r * np.sin(theta - theta0)
    x = r * np.cos(theta - theta0)
    sigma = 0.7 * np.exp(r_max / (x**2 + r_max))
    factor = 1.5
    dy_dr = -2 * M * y**2 * np.exp(2 * M * (r - r0))
    dx_dr = np.cos(theta - theta0)
    dz_dr = np.sin(theta - theta0)
    dsigma_dr = -2 * x * sigma * (r_max / (x**2 + r_max)**2) * dx_dr
    n_e = 100 * (n_p0 / r) * np.exp(-(factor * z**2) /
                                    sigma**2) * (1 - y) + n_p0 / r
    part1 = (-np0 / r**2) * (1 + 100 *
                             np.exp(-(factor * z**2) / sigma**2) * (1 - y))
    part2 = -(np0 / r) * 100 * np.exp(-(factor * z**2) / sigma**2) * (dy_dr + 2 * \
              factor * (1 - y) * (z / sigma) * ((dz_dr / sigma) - (z / sigma**2) * dsigma_dr))
    return part1 + part2


def find_normal_to_constant_n_surface(np0, r, theta, phi, r_max):
    dn_dr = delta_n_delta_r(np0, r, theta, phi, r_max)
    dn_dtheta = prop_tool_solve_int.delta_n_delta_theta(
        n_p0, r, theta, phi, r_max)
    dn_dphi = prop_tool_solve_int.delta_n_delta_phi(n_p0, r, theta, phi, r_max)
    return np.array([dn_dr, dn_dtheta / r, dn_dphi / (r * np.sin(theta))])


def get_angle_bet_k_n(np0, r, theta, phi, theta_dash, phi_dash, r_max):
    n_normal_vect = find_normal_to_constant_n_surface(
        np0, r, theta, phi, r_max)
    k_vect = np.array([1, r * theta_dash, r * np.sin(theta) * phi_dash])
    n_normal = prop_tool_ini_cond.get_vector_modulus(n_normal_vect)
    k = prop_tool_ini_cond.get_vector_modulus(k_vect)
    return np.arccos(np.dot(n_normal_vect, k_vect) / (n_normal * k))


star_name = 'hd133880'
alpha, beta = 55.0 * (np.pi / 180.0), 78 * (np.pi / 180.0)
Bp = 9600.0
n_p0 = 10**9
mode = 1  # 0 means O-mode and 1 means X-mode
harm_no = 2.01  # harmonic number, must be >1 for X mode
R_A = 60.0
L = 71.0
nu_arr = [400]
rho_fact    =100.0

rot_phase = np.linspace(0.5, 1.5, 1000)

sigma_theta = 3 * (np.pi / 180.)
B_curve = np.zeros(len(rot_phase))

ax = plt.subplot(len(nu_arr), 1, 1)

for num in range(len(nu_arr)):
    nu  =nu_arr[num]
    ax = plt.subplot(len(nu_arr), 1, num+1)
    n_ray_path, s_ray_path = [], []
    if mode == 1:
        out_filename = 'cython_X_mode_' + \
            str(int(nu)) + 'MHz_density_RRM_' + star_name + '_like_rho_fact_' + str(rho_fact) + '.p'
    if mode == 0:
        out_filename = 'cython_O_mode_' + \
            str(int(nu)) + 'MHz_density_RRM_' + star_name + '_like_rho_fact_' + str(rho_fact) + '.p'

    output = pickle.load(open(out_filename, 'rb'))
    n_ray_path = output['N_ray_path']
    s_ray_path = output['S_ray_path']
    
    n_k_out_cart, s_k_out_cart = [], []
    n_k_out_mod, s_k_out_mod = np.zeros(
        len(n_ray_path)), np.zeros(
        len(s_ray_path))

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

    n_flux, s_flux = np.zeros(len(rot_phase)), np.zeros(len(rot_phase))
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
    print(max_flux,nu)
    plt.plot(rot_phase, n_flux / max_flux, 'r-',
             lw=1, label='lightcurve: north pole')
    plt.plot(rot_phase, s_flux / max_flux, 'b-',
             lw=1, label='lightcurve: south pole')
    plt.plot(rot_phase, B_curve, 'k--', lw=2, label=r'$B_\mathrm{LoS}$')
    plt.grid()
    plt.legend(loc='best')
    #plt.show()
plt.xlabel('Rotational Phase')

plt.tight_layout()

plt.show()
