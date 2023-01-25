cdef test_func(double r,double [:]full_y_arr,double [:]theta_phi_dash0,double n_p0,float Bp,float nu,int mode,float r_max,double rho_fact):
    theta,phi,Y1,Y2 =full_y_arr[0],full_y_arr[1],full_y_arr[2],full_y_arr[3]
    Y_arr		=np.array([Y1,Y2])
    ini_guess           =np.array([theta_phi_dash0[0],theta_phi_dash0[1]])
    ans		        =optimize.root(minimize_func,x0=ini_guess,args=(r,theta,phi,Y_arr,n_p0,Bp,nu,mode,r_max,rho_fact))		
    if ans['success']==True:
        theta_phi_dash0[0]	=ans['x'][0]
        theta_phi_dash0[1]	=ans['x'][1]
        dtheta_dr  =theta_phi_dash0[0]
        dphi_dr  =theta_phi_dash0[1]
        cos_psi		=c_get_cos_ang_bet_prop_vect_B_vect(r,theta,phi,dtheta_dr,dphi_dr)
        nu_p		=get_plasma_freq(n_p0,r,theta,phi,r_max,rho_fact)
	nu_B		=c_get_cyclotron_freq(r,theta,Bp)
	sigma		=get_sigma(nu,nu_p,nu_B,cos_psi)
	tau		=get_tau(sigma,cos_psi,nu_p,nu)
	mu		=get_refractive_index(nu_p,nu_B,tau,cos_psi,nu,mode)
        G		=G_func(r,theta,dtheta_dr,dphi_dr)
	dy1_dr	        =delta_F_delta_theta(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,dtheta_dr,dphi_dr,n_p0,r_max,rho_fact)
        dy2_dr	        =delta_F_delta_phi(mu,G,cos_psi,nu_p,nu_B,sigma,tau,nu,mode,r,theta,phi,dtheta_dr,dphi_dr,n_p0,r_max,rho_fact)
        return np.array([dtheta_dr,dphi_dr,dy1_dr,dy2_dr])

    else:
        print('theta_dash and phi_dash could not be solved')
        raise KeyboardInterrupt

