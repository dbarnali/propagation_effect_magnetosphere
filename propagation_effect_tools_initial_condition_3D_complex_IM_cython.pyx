# This code finds out the initial condition to find out the ray path inside the inner magnetosphere, given a frequency nu, harmonic number
# harm_no and polar field strength Bp, assuming that the ray is originated along a field line given by r=Lsin^2theta
# The initial conditions are the starting values of the variables inside
# the inner magnetosphere


from libc.string cimport memcpy
import propagation_effect_tools_density_func as prop_tool_density
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


np.import_array()
cdef double pi
pi = acos(-1.0)


cdef extern from 'gsl/gsl_poly.h':
    int gsl_poly_solve_cubic(double a, double b, double c, double * x0, double * x1, double * x2)

cdef double get_B_from_nu(float nu, float harm_no):  # nu in MHz, B in gauss
    return nu / (2.8 * harm_no)

cdef double func1(double theta, double R_A, double r01, double theta01):
    return R_A * sin(theta)**2 * cos(theta) - r01 * cos(theta01)

cdef double func2(double theta, double L, double Bp, double B):  # nu_nuB=nu_B/nu_Bp=B/Bp
    cdef double nu_nuB = B / Bp
    return L**3 * nu_nuB - sqrt(1 - (3 / 4.0) * sin(theta)**2) / sin(theta)**6.

cdef double func3(double phi, double r0, double theta0, double r01, double theta01, double phi01, double kx, double ky):
    cdef double a = ky * (r0 * sin(theta0) * cos(phi) - r01 * sin(theta01) * cos(phi01))
    cdef double b = kx * (r0 * sin(theta0) * sin(phi) - r01 * sin(theta01) * sin(phi01))
    return abs(a + b)


cdef void c_get_sin_phi0(double r0, double theta0, double r01, double theta01, double phi01, double kx, double ky, double * x1, double * x2):
    cdef double alpha, beta, gamma, temp, den
    alpha = r0 * ky * sin(theta0)
    beta = r0 * kx * sin(theta0)
    gamma = r01 * ky * sin(theta01) * cos(phi01) - r01 * \
        kx * sin(theta01) * sin(phi01)
    temp = alpha * sqrt(alpha**2 + beta**2 - gamma**2)
    den = alpha**2 + beta**2
    x1[0] = ((-gamma * beta) + temp) / den
    x2[0] = ((-gamma * beta) - temp) / den
    return

cdef void c_B_func(double r, double theta, double Bp, double * B_sph):
    B_sph[0] = Bp * (cos(theta)) / r**3
    B_sph[1] = Bp * (0.5 * sin(theta)) / r**3
    B_sph[2] = 0.0
    return

# here V is an array with components are the r,theta,phi components of vector V
cpdef convert_spherical_to_cartesian_coord(np.ndarray[np.double_t, ndim=1] V, double r, double theta, double phi):
    cdef double V_r, V_theta, V_phi
    V_r, V_theta, V_phi = V[0], V[1], V[2]
    cdef double V_x = V_r * sin(theta) * cos(phi) + V_theta * cos(theta) * cos(phi) - V_phi * sin(phi)
    cdef double V_y = V_r * sin(theta) * sin(phi) + V_theta * cos(theta) * sin(phi) + V_phi * cos(phi)
    cdef double V_z = V_r * cos(theta) - V_theta * sin(theta)
    return np.array([V_x, V_y, V_z])

# here V is an array with components are the x,y,z components of vector V
cpdef convert_cartesian_to_spherical_coord(np.ndarray[np.double_t, ndim=1] V, double r, double theta, double phi):
    cdef double V_x, V_y, V_z
    V_x, V_y, V_z = V[0], V[1], V[2]
    cdef double V_r = V_x * sin(theta) * cos(phi) + V_y * sin(theta) * sin(phi) + V_z * cos(theta)
    cdef double V_theta = V_x * cos(theta) * cos(phi) + V_y * cos(theta) * sin(phi) - V_z * sin(theta)
    cdef double V_phi = -V_x * sin(phi) + V_y * cos(phi)
    return np.array([V_r, V_theta, V_phi])

cdef void c_convert_spherical_to_cartesian_coord(double * V, double r, double theta, double phi, double * V_cart):
    cdef double V_r, V_theta, V_phi
    V_r = V[0]
    V_theta = V[1]
    V_phi = V[2]
    V_cart[0] = V_r * sin(theta) * cos(phi) + V_theta * \
        cos(theta) * cos(phi) - V_phi * sin(phi)
    V_cart[1] = V_r * sin(theta) * sin(phi) + V_theta * \
        cos(theta) * sin(phi) + V_phi * cos(phi)
    V_cart[2] = V_r * cos(theta) - V_theta * sin(theta)
    return

cdef void c_convert_cartesian_to_spherical_coord(double * V, double r, double theta, double phi, double * V_sph):
    cdef double V_x, V_y, V_z
    V_x, V_y, V_z = V[0], V[1], V[2]
    V_sph[0] = V_x * sin(theta) * cos(phi) + V_y * \
        sin(theta) * sin(phi) + V_z * cos(theta)
    V_sph[1] = V_x * cos(theta) * cos(phi) + V_y * \
        cos(theta) * sin(phi) - V_z * sin(theta)
    V_sph[2] = -V_x * sin(phi) + V_y * cos(phi)
    return

cdef double c_get_vector_modulus(double * V, int len_arr):
    cdef double tot = 0.0
    cdef int i
    for i in range(len_arr):
        tot += V[i] * V[i]
    return sqrt(tot)

cpdef get_vector_modulus(np.ndarray[np.double_t, ndim=1] V):
    return sqrt(np.sum(V**2))

cdef double c_get_cyclotron_freq(double r, double theta, float Bp):
    cdef double B
    cdef double B_vect[3]
    c_B_func(r, theta, Bp, B_vect)
    B = c_get_vector_modulus(B_vect, 3)
    return 2.8 * B  # in MHz

# already cythonized
cdef double get_plasma_freq(double n_p0, double r, double theta, double phi, float r_max, float rho_fact):
    # print(r,theta,phi,'-------------------')
    # in cgs
    cdef double n_e = prop_tool_density.density_func(n_p0, r, theta, phi, r_max, rho_fact)
    return 9.0 * 10.**(-3) * sqrt(n_e)  # in MHz


cpdef get_normal_to_IM_boundary(double theta, double phi):
    cdef float n_r = 1.
    cdef double n_theta = -2. / tan(theta)
    cdef float n_phi = 0.0
    return np.array([n_r, n_theta, n_phi])


def get_refraction_angle(double mu1, double mu2, double inc_ang):
    if (mu1 * sin(inc_ang)) / mu2 > 1 or np.isnan(mu2) or np.isnan(inc_ang):
        return False, 0
    else:
        return True, asin((mu1 * sin(inc_ang)) / mu2)


cdef int c_get_refraction_angle(double mu1, double mu2, double inc_ang, double * n_theta_r):
    if (mu1 * sin(inc_ang)) / mu2 <= 1:
        n_theta_r[0] = asin((mu1 * sin(inc_ang)) / mu2)
        return 1
    else:
        return 0

# in spherical polar coordinates
cpdef get_k_vector_inside_IM_polar(double r, double theta, double phi, double theta_dash, double phi_dash):
    return np.array([1., r * theta_dash, r * sin(theta) * phi_dash])

cpdef func4(np.ndarray[np.double_t, ndim=1] theta_phi_dash, double r, double theta, double phi, np.ndarray[np.double_t, ndim=1] k_01_polar, np.ndarray[np.double_t, ndim=1] normal_vect_polar, double inc_ang, double theta_r):
    cdef double theta_dash, phi_dash, k01, k0, normal_vect, v1, v2
    cdef np.ndarray[np.double_t, ndim = 1] k_0_polar
    theta_dash, phi_dash = theta_phi_dash[0], theta_phi_dash[1]
    k_0_polar = get_k_vector_inside_IM_polar(
        r, theta, phi, theta_dash, phi_dash)
    k_01, k_0 = get_vector_modulus(k_01_polar), get_vector_modulus(k_0_polar)
    normal_vect = get_vector_modulus(normal_vect_polar)
    v1 = np.dot(k_01_polar, k_0_polar) - k_01 * k_0 * cos(theta_r - inc_ang)
    v2 = np.dot(normal_vect_polar, k_0_polar) + \
        normal_vect * k_0 * cos(theta_r)
    return np.array([v1, v2])

cpdef get_theta_dash0_phi_dash0(double r, double theta, double phi, np.ndarray[np.double_t, ndim=1] k_01_polar, np.ndarray[np.double_t, ndim=1] normal_vect_polar, double inc_ang, double theta_r, double tol=1e-6):
    cdef double k_r, k_theta, k_phi, n_r, n_theta, k, n, A, B, den, C, D, denom, numer, theta_dash0, phi_dash0
    k_r, k_theta, k_phi = k_01_polar[0], k_01_polar[1], k_01_polar[2]
    n_r, n_theta = normal_vect_polar[0], normal_vect_polar[1]
    k = get_vector_modulus(k_01_polar)
    n = get_vector_modulus(normal_vect_polar)
    A, B = -n_r / (n * cos(theta_r)), -(r * n_theta) / (n * cos(theta_r))
    den = r * k_phi * sin(theta)
    C, D = (B * k * cos(theta_r - inc_ang) - r * k_theta) / \
        den, (A * k * cos(theta_r - inc_ang) - k_r) / den
    denom = r * (-n_r * k_phi + C * sin(theta) *
                 (n_r * k_theta - n_theta * k_r))
    numer = -n_theta * k_phi - r * D * \
        sin(theta) * (n_r * k_theta - n_theta * k_r)
    theta_dash0 = numer / denom
    phi_dash0 = C * theta_dash0 + D
    return theta_dash0, phi_dash0


cdef int c_check_soln_validity(double * k_01, double r0, double theta0, double phi0, double r01, double theta01, double phi01, double tol):
    cdef double kx, ky, X1, X2
    kx = k_01[0]
    ky = k_01[1]
    X1 = ky * (r0 * sin(theta0) * cos(phi0) - r01 * sin(theta01) * cos(phi01))
    X2 = kx * (r0 * sin(theta0) * sin(phi0) - r01 * sin(theta01) * sin(phi01))
    if abs(X1 - X2) < tol:
        return 1
    else:
        return 0

cdef double c_get_cos_ang_bet_prop_vect_B_vect(double r, double theta, double phi, double theta_dash, double phi_dash):
    cdef double k_vect_polar[3]
    cdef double B_vect_polar[3]
    cdef double k, B
    k_vect_polar[:] = [1., r * theta_dash, r * sin(theta) * phi_dash]
    c_B_func(r, theta, 1.0, B_vect_polar)
    k = c_get_vector_modulus(k_vect_polar, 3)
    B = c_get_vector_modulus(B_vect_polar, 3)
    return calc_dot_product(k_vect_polar, B_vect_polar, 3) / (k * B)


cdef double calc_dot_product(double * v1, double * v2, len_arr):
    cdef double tot = 0.0
    cdef int i
    for i in range(len_arr):
        tot += v1[i] * v2[i]
    return tot

cdef void c_get_normal_to_IM_boundary(double theta, double phi, double * n_sph):
    n_sph[0] = 1.
    n_sph[1] = -2. / tan(theta)
    n_sph[2] = 0.0
    return

cdef void c_get_theta_dash0_phi_dash0(double r, double theta, double phi, double * k_01_polar, double * normal_vect_polar, double inc_ang, double theta_r, double tol, double * theta_dash0, double * phi_dash0):
    cdef double k_r, k_theta, k_phi, n_r, n_theta, k, n, A, B, den, C, D, denom, numer
    k_r = k_01_polar[0]
    k_theta = k_01_polar[1]
    k_phi = k_01_polar[2]
    n_r = normal_vect_polar[0]
    n_theta = normal_vect_polar[1]
    k = c_get_vector_modulus(k_01_polar, 3)
    n = c_get_vector_modulus(normal_vect_polar, 3)
    A = -n_r / (n * cos(theta_r))
    B = -(r * n_theta) / (n * cos(theta_r))
    den = r * k_phi * sin(theta)
    C = (B * k * cos(theta_r - inc_ang) - r * k_theta) / den
    D = (A * k * cos(theta_r - inc_ang) - k_r) / den
    denom = r * (-n_r * k_phi + C * sin(theta) *
                 (n_r * k_theta - n_theta * k_r))
    numer = -n_theta * k_phi - r * D * \
        sin(theta) * (n_r * k_theta - n_theta * k_r)
    theta_dash0[0] = numer / denom
    phi_dash0[0] = C * theta_dash0[0] + D
    return

cdef c_find_mu_in(double mu, double inc_ang, double r0, double theta0, double phi0, float nu, double nu_p0, double nu_B0, double * k_01_polar, double * normal_vect_polar, float R_A, int mode):
    cdef double theta_r, theta_dash0, phi_dash0, cos_psi0, sigma0, tau0, new_mu
    theta_r = asin(sin(inc_ang) / mu)
    c_get_theta_dash0_phi_dash0(r0, theta0, phi0, k_01_polar, normal_vect_polar, inc_ang, theta_r, 1.0e-6, & theta_dash0, & phi_dash0)
    cos_psi0 = c_get_cos_ang_bet_prop_vect_B_vect(
        r0, theta0, phi0, theta_dash0, phi_dash0)
    sigma0 = get_sigma(nu, nu_p0, nu_B0, cos_psi0)
    tau0 = get_tau(sigma0, cos_psi0, nu_p0, nu)
    new_mu = get_refractive_index(nu_p0, nu_B0, tau0, cos_psi0, nu, mode)
    return abs(mu - new_mu)

cdef find_mu_in(double mu, double inc_ang, double r0, double theta0, double phi0, float nu, double nu_p0, double nu_B0, np.ndarray[np.double_t, ndim=1] k_01_polar, np.ndarray[np.double_t, ndim=1] normal_vect_polar, float R_A, int mode):
    cdef double theta_r, theta_dash0, phi_dash0, cos_psi0, sigma0, tau0, new_mu
    theta_r = asin(sin(inc_ang) / mu)
    theta_dash0, phi_dash0 = get_theta_dash0_phi_dash0(
        r0, theta0, phi0, k_01_polar, normal_vect_polar, inc_ang, theta_r)
    cos_psi0 = get_cos_ang_bet_prop_vect_B_vect(
        r0, theta0, phi0, theta_dash0, phi_dash0)
    sigma0 = get_sigma(nu, nu_p0, nu_B0, cos_psi0)
    tau0 = get_tau(sigma0, cos_psi0, nu_p0, nu)
    new_mu = get_refractive_index(nu_p0, nu_B0, tau0, cos_psi0, nu, mode)
    return np.abs(mu - new_mu)

cpdef B_func(double r, double theta, double Bp):
    cdef double Br, Bt
    Br, Bt = (cos(theta)) / r**3, (0.5 * sin(theta)) / r**3
    return np.array([Br * Bp, Bt * Bp, 0.])

# alternate function written
cpdef get_cos_ang_bet_prop_vect_B_vect(double r, double theta, double phi, double theta_dash, double phi_dash):
    cdef np.ndarray[np.double_t, ndim = 1] k_vect_volar, B_vect_polar
    cdef double k, B
    k_vect_polar = np.array([1., r * theta_dash, r * sin(theta) * phi_dash])
    B_vect_polar = B_func(r, theta, 1.0)
    k, B = get_vector_modulus(k_vect_polar), get_vector_modulus(B_vect_polar)
    return np.dot(k_vect_polar, B_vect_polar) / (k * B)

cdef double get_sigma(float nu, double nu_p, double nu_B, double cos_psi):  # already cythonized
    cdef double factor
    factor = 0.5 * nu / abs(nu**2 - nu_p**2)
    return factor * nu_B * (1 - cos_psi**2)

cdef double get_tau(double sigma, double cos_psi, double nu_p, float nu):  # already cythonized
    if nu > nu_p:
        return -(sigma + sqrt(sigma**2 + cos_psi**2))
    else:
        return (sigma + sqrt(sigma**2 + cos_psi**2))


# already cythonized
cdef double get_refractive_index(double nu_p, double nu_B, double tau, double cos_psi, float nu, int mode):
    if mode == 1:  # X-mode
        return sqrt(1 - (nu_p**2 / (nu * (nu + tau * nu_B))))
    else:		# O-mode
        return sqrt(
            1 - ((tau * nu_p**2) / (nu * (tau * nu - nu_B * cos_psi**2))))


cdef double min_1darr(double * arr, int len_arr):
    cdef double min_val
    cdef int i
    min_val = arr[0]
    for i in range(len_arr - 1):
        if min_val > arr[i + 1]:
            min_val = arr[i + 1]
    return min_val

cdef void c_get_initial_condition_array(double n_p0, float R_A, double nu_B0, float nu, int mode, double r0, double theta0, double phi0,
                                        double * k_01, double cos_psi, double * n_ini_array, double * s_ini_array, int * n_is_true, int * s_is_true, float rho_fact):
    cdef np.ndarray[np.double_t, ndim = 1] np_k_01_polar, np_normal_vect_polar
    cdef double inc_ang, mu_1, nu_p0, sigma0, tau0, mu_2n, mu_2s, n_theta_r, mu_2, s_theta_r, theta_dash0, phi_dash0
    cdef double normal_vect_polar[3]
    cdef double normal_vect_cart[3]
    cdef double k_01_polar[3]
    cdef double k_IM[3]

    c_get_normal_to_IM_boundary(theta0, phi0, normal_vect_polar)
    c_convert_spherical_to_cartesian_coord(
        normal_vect_polar, r0, theta0, phi0, normal_vect_cart)
    inc_ang = pi - acos(calc_dot_product(k_01,
                                         normal_vect_cart,
                                         3) / (c_get_vector_modulus(k_01,
                                                                    3) * c_get_vector_modulus(normal_vect_cart,
                                                                                              3)))
    mu_1 = 1.0
    nu_p0 = get_plasma_freq(n_p0, r0, theta0, phi0, R_A, rho_fact)
    #print(nu_p0 , 'nu_p0')
    
    sigma0 = get_sigma(nu, nu_p0, nu_B0, cos_psi)
    tau0 = get_tau(sigma0, cos_psi, nu_p0, nu)
    mu_2n = get_refractive_index(nu_p0, nu_B0, tau0, cos_psi, nu, mode)
    #print(mu_2n ,inc_ang*180/np.pi, 'mu_2n, inc_ang')    
    n_is_true[0] = c_get_refraction_angle(mu_1, mu_2n, inc_ang, & n_theta_r)
    #print(n_is_true[0] , 'n_is_true[0]')
    # CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    if n_is_true[0] == 1:
        c_convert_cartesian_to_spherical_coord(
            k_01, r0, theta0, phi0, k_01_polar)

        np_k_01_polar = np.array([k_01_polar[0], k_01_polar[1], k_01_polar[2]])
        np_normal_vect_polar = np.array(
            [normal_vect_polar[0], normal_vect_polar[1], normal_vect_polar[2]])

        ans = optimize.root(
            find_mu_in,
            x0=mu_2n,
            args=(
                inc_ang,
                r0,
                theta0,
                phi0,
                nu,
                nu_p0,
                nu_B0,
                np_k_01_polar,
                np_normal_vect_polar,
                R_A,
                mode))
        if ans['success']:
            mu_2 = ans['x'][0]
            n_theta_r = asin(sin(inc_ang) / mu_2)
            c_get_theta_dash0_phi_dash0(r0, theta0, phi0, k_01_polar, normal_vect_polar, inc_ang, n_theta_r, 1.0e-6, & theta_dash0, & phi_dash0)
            n_ini_array[:] = [r0, theta0, phi0, theta_dash0, phi_dash0]
            k_IM[:] = [1, r0 * theta_dash0, r0 * sin(theta0) * phi_dash0]
        else:
            n_is_true[0] = 0
            n_ini_array[:] = [0., 0., 0., 0., 0.]
    else:
        n_ini_array[:] = [0., 0., 0., 0., 0.]

    # SOUTH POLE

    nu_p0 = get_plasma_freq(n_p0, r0, pi - theta0, phi0, R_A, rho_fact)
    sigma0 = get_sigma(nu, nu_p0, nu_B0, cos_psi)
    tau0 = get_tau(sigma0, cos_psi, nu_p0, nu)
    mu_2s = get_refractive_index(nu_p0, nu_B0, tau0, cos_psi, nu, mode)
    s_is_true[0] = c_get_refraction_angle(mu_1, mu_2s, inc_ang, & s_theta_r)

    if s_is_true[0] == 1:
        c_get_normal_to_IM_boundary(pi - theta0, phi0, normal_vect_polar)
        c_convert_spherical_to_cartesian_coord(
            normal_vect_polar, r0, theta0, phi0, normal_vect_cart)
        c_convert_cartesian_to_spherical_coord(
            k_01, r0, pi - theta0, phi0, k_01_polar)

        np_k_01_polar = np.array([k_01_polar[0], k_01_polar[1], k_01_polar[2]])
        np_normal_vect_polar = np.array(
            [normal_vect_polar[0], normal_vect_polar[1], normal_vect_polar[2]])

        ans = optimize.root(
            find_mu_in,
            x0=mu_2s,
            args=(
                inc_ang,
                r0,
                pi - theta0,
                phi0,
                nu,
                nu_p0,
                nu_B0,
                np_k_01_polar,
                np_normal_vect_polar,
                R_A,
                mode))
        if ans['success']:
            mu_2 = ans['x'][0]
            s_theta_r = asin(sin(inc_ang) / mu_2)
            c_get_theta_dash0_phi_dash0(r0, pi - theta0, phi0, k_01_polar, normal_vect_polar, inc_ang, s_theta_r, 1.0e-6, & theta_dash0, & phi_dash0)
            s_ini_array[:] = [r0, pi - theta0, phi0, theta_dash0, phi_dash0]
            k_IM[:] = [-1., -r0 * theta_dash0, -r0 * sin(theta0) * phi_dash0]
        else:
            s_is_true[0] = 0
            s_ini_array[:] = [0., 0., 0., 0., 0.]

    else:
        s_ini_array[:] = [0., 0., 0., 0., 0.]
    return


cpdef get_initial_condition(float nu, float harm_no, float Bp, float L, float R_A, float phi01, double n_p0, int mode, float rho_fact):
    #print('***************INITIAL CONDITION')
    cdef double B, h, ini_guess01, theta01, r01, theta0, r0, phi0, sin_phi01, nu_B0, phi0_1, phi0_2, cos_psi
    cdef double tol
    cdef int pos, pos1, i, j
    cdef int n_is_true1, n_is_true2, s_is_true1, s_is_true2
    cdef double p[3]
    cdef double cos_theta0[3]
    cdef double B_polar[3]
    cdef double B_cart[3]
    cdef double k_01[3]
    cdef double m_k_01[3]
    cdef double B0_polar[3]
    cdef double B0_cart[3]
    cdef double a, b, c, sin_phi0_1, sin_phi0_2
    cdef int num = 0
    cdef double * temp
    cdef double * theta0_arr
    cdef double n_ini_arr1[5]
    cdef double n_ini_arr2[5]
    cdef double s_ini_arr1[5]
    cdef double s_ini_arr2[5]
    cdef np.ndarray[np.double_t, ndim= 1] np_n_ini_arr1, np_n_ini_arr2, np_s_ini_arr1, np_s_ini_arr2

    B = get_B_from_nu(nu, harm_no)  # B in gauss, nu in MHz
    h = (Bp / B)**(1 / 3.0)
    ini_guess01 = asin(sqrt(h / L))
    ans = optimize.root(func2, x0=ini_guess01, args=(L, Bp, B))

    if ans['success']:
        theta01 = ans['x'][0]
    else:
        print(r'For nu=' + str(nu) + ' MHz, R_A=' + str(R_A) + ', L=' +
              str(L) + ',theta01 cannotbe found\nchange initial guess')
        raise KeyboardInterrupt

    
    r01 = L * sin(theta01)**2
    p[:] = [0., -1.0, (r01 * cos(theta01)) / R_A]
    # p	=np.array([1.0,0,-1.0,(r01*cos(theta01))/R_A])
    pos1 = gsl_poly_solve_cubic(p[0], p[1], p[2], & a, & b, & c)

    cos_theta0[:] = [a, b, c]  # np.roots(p)

    if abs(a) > 1:
        num += 1
    if abs(b) > 1:
        num += 1
    if abs(c) > 1:
        num += 1

    temp = <double * >PyMem_Malloc((3 - num) * sizeof(double))
    theta0_arr = <double * >PyMem_Malloc((3 - num) * sizeof(double))

    j = 0
    for i in range(3):
        if abs(cos_theta0[i]) <= 1:
            temp[j] = cos_theta0[i]
            j += 1

    for i in range(3 - num):
        theta0_arr[i] = acos(temp[i])

    theta0 = min_1darr(theta0_arr, 3 - num)

    c_B_func(r01, theta01, Bp, B_polar)
    c_convert_spherical_to_cartesian_coord(
        B_polar, r01, theta01, phi01, B_cart)
    k_01[:] = [B_cart[1], -B_cart[0], 0.0]
    r0 = R_A * sin(theta0)**2
    c_get_sin_phi0(r0, theta0, r01, theta01, phi01, k_01[0], k_01[1], & sin_phi0_1, & sin_phi0_2)
    c_B_func(r0, theta0, Bp, B0_polar)
    nu_B0 = c_get_cyclotron_freq(r0, theta0, Bp)

    if (r0 * sin(theta0) * sin_phi0_1 - r01 *
            sin(theta01) * sin(phi01)) * k_01[1] > 0:
        phi0_1 = asin(sin_phi0_1)
        phi0_2 = pi - asin(sin_phi0_1)
        phi0 = 9999999
        if c_check_soln_validity(
                k_01,
                r0,
                theta0,
                phi0_1,
                r01,
                theta01,
                phi01,
                1.e-6) == 1:
            phi0 = phi0_1
        else:
            if c_check_soln_validity(
                    k_01,
                    r0,
                    theta0,
                    phi0_2,
                    r01,
                    theta01,
                    phi01,
                    tol=1e-6) == 1:
                phi0 = phi0_2
        if phi0 == 9999999:
            raise RuntimeError("Phi0 could not be solved. Please check\n")

    else:
        phi0_1 = asin(sin_phi0_2)
        phi0_2 = pi - asin(sin_phi0_2)
        phi0 = 9999999
        if c_check_soln_validity(
                k_01,
                r0,
                theta0,
                phi0_1,
                r01,
                theta01,
                phi01,
                tol=1e-6) == 1:
            phi0 = phi0_1
        else:
            if c_check_soln_validity(
                    k_01,
                    r0,
                    theta0,
                    phi0_2,
                    r01,
                    theta01,
                    phi01,
                    tol=1e-6) == 1:
                phi0 = phi0_2
        if phi0 == 9999999:
            raise RuntimeError("Phi0 could not be solved. Please check\n")
    #print(r0,theta0*180/np.pi,phi0 * 180 / np.pi, 'r0,theta0,phi0')
    c_convert_spherical_to_cartesian_coord(B0_polar, r0, theta0, phi0, B0_cart)
    cos_psi = calc_dot_product(k_01,
                               B0_cart,
                               3) / (c_get_vector_modulus(k_01,
                                                          3) * c_get_vector_modulus(B0_cart,
                                                                                    3))
    phi0_1 = phi0
    phi0_2 = 2 * phi01 - phi0

    c_get_initial_condition_array(n_p0, R_A, nu_B0, nu, mode, r0, theta0, phi0, k_01, cos_psi, n_ini_arr1, s_ini_arr1, & n_is_true1, & s_is_true1, rho_fact)
    m_k_01[:] = [-k_01[0], -k_01[1], -k_01[2]]

    c_get_initial_condition_array(n_p0, R_A, nu_B0, nu, mode, r0, theta0, 2 * phi01 - phi0, m_k_01, cos_psi, n_ini_arr2, s_ini_arr2,
                                  & n_is_true2, & s_is_true2, rho_fact)
    # **IMP: we have used the azimuthal symmetry of the magnetic field by assuming that psi does not change
    # for the oppositely travelling rays

    np_n_ini_arr1 = np.array(
        [n_ini_arr1[0], n_ini_arr1[1], n_ini_arr1[2], n_ini_arr1[3], n_ini_arr1[4]])
    np_n_ini_arr2 = np.array(
        [n_ini_arr2[0], n_ini_arr2[1], n_ini_arr2[2], n_ini_arr2[3], n_ini_arr2[4]])
    np_s_ini_arr1 = np.array(
        [s_ini_arr1[0], s_ini_arr1[1], s_ini_arr1[2], s_ini_arr1[3], s_ini_arr1[4]])
    np_s_ini_arr2 = np.array(
        [s_ini_arr2[0], s_ini_arr2[1], s_ini_arr2[2], s_ini_arr2[3], s_ini_arr2[4]])

    PyMem_Free(temp)
    PyMem_Free(theta0_arr)
    return [n_is_true1, n_is_true2, s_is_true1, s_is_true2], [
        np_n_ini_arr1, np_n_ini_arr2, np_s_ini_arr1, np_s_ini_arr2]
