import numpy as np

GAMMA=1.4

def Flux(in_U_mat):
    # 1D Euler equation
    F_mat=np.zeros(in_U_mat.shape)
    F_mat[:,0]=in_U_mat[:,1]
    RhoU2_arr=in_U_mat[:,1]**2/in_U_mat[:,0]
    p_arr=(in_U_mat[:,2]-0.5*RhoU2_arr)*(GAMMA-1)
    F_mat[:,1]=RhoU2_arr+p_arr
    F_mat[:,2]=(in_U_mat[:,2]+p_arr)*in_U_mat[:,1]/in_U_mat[:,0]
    return F_mat

def V2U_arr(in_V_arr):
    U_arr=np.zeros(in_V_arr.shape)
    U_arr[0]=in_V_arr[0]
    U_arr[1]=in_V_arr[0]*in_V_arr[1]
    U_arr[2]=in_V_arr[2]/(GAMMA-1)+0.5*in_V_arr[0]*in_V_arr[1]**2
    return U_arr

def V2U_mat(in_V_mat):
    U_mat=np.zeros(in_V_mat.shape)
    U_mat[:,0]=in_V_mat[:,0]
    U_mat[:,1]=in_V_mat[:,0]*in_V_mat[:,1]
    U_mat[:,2]=in_V_mat[:,2]/(GAMMA-1)+0.5*U_mat[:,1]*in_V_mat[:,1]
    return U_mat

def U2V_mat(in_U_mat):
    V_mat=np.zeros(in_U_mat.shape)
    V_mat[:,0]=in_U_mat[:,0]
    V_mat[:,1]=in_U_mat[:,1]/V_mat[:,0]
    V_mat[:,2]=(in_U_mat[:,2]-0.5*V_mat[:,0]*V_mat[:,1]**2)*(GAMMA-1)
    return V_mat

def IC(in_x_arr,in_type):
    if(in_type==1):
        #  idx_l=in_x_arr<=0.5
        #  idx_h=in_x_arr>0.5
        idx_l=in_x_arr<=0.0
        idx_h=in_x_arr>0.0
        V_mat=np.zeros((in_x_arr.size,3))
        V_mat[idx_l,0]=rho_l_init
        V_mat[idx_l,1]=u_l_init
        V_mat[idx_l,2]=p_l_init
        V_mat[idx_h,0]=rho_r_init
        V_mat[idx_h,1]=u_r_init
        V_mat[idx_h,2]=p_r_init
        U_mat=V2U_mat(V_mat)
    else:
        exit("Not implemented!")
    return U_mat

def RK1(in_U_mat,in_dt,in_dx,in_limiter_type):
    dF_mat=Res(in_U_mat,in_dx,in_limiter_type)
    U_new_mat=in_U_mat-in_dt*dF_mat
    return U_new_mat

def RK2(in_U_mat,in_dt,in_dx,in_order,in_doLimiting,in_limiter_type,in_FluxType,in_isAUSMPlus):
    from numpy import copy
    U_old_mat=copy(in_U_mat)
    dF_mat=Res(in_U_mat,in_dx,in_order,in_doLimiting,in_limiter_type,in_FluxType,in_isAUSMPlus)
    in_U_mat=in_U_mat-in_dt*dF_mat
    dF_mat=Res(in_U_mat,in_dx,in_order,in_doLimiting,in_limiter_type,in_FluxType,in_isAUSMPlus)
    in_U_mat=in_U_mat-in_dt*dF_mat
    U_new_mat=0.5*(U_old_mat+in_U_mat)
    return U_new_mat

def Res(in_U_cell_mat,in_dx,in_order,in_doLimiting,in_limiter_type,in_FluxType,in_isAUSMPlus):
    # Since t_end=0.2 when the flow interface does not reach both ends, BCs are not important.
    # Fix all
    # Fixed BC for shock tube
    #  V_l_arr=np.array([rho_l_init,u_l_init,p_l_init])
    #  U_l_arr=V2U_arr(V_l_arr)
    #  V_r_arr=np.array([rho_r_init,u_r_init,p_r_init])
    #  U_r_arr=V2U_arr(V_r_arr)
    #  U_cell_ghost_mat=np.append(np.reshape(U_l_arr,(1,3)),in_U_cell_mat,axis=0)
    #  U_cell_ghost_mat=np.append(U_cell_ghost_mat,np.reshape(U_r_arr,(1,3)),axis=0)
    # BC: Extrapolate solution variables at both ends
    U_cell_ghost_mat=np.append(np.reshape(in_U_cell_mat[0,:],(1,3)),in_U_cell_mat,axis=0)
    U_cell_ghost_mat=np.append(U_cell_ghost_mat,np.reshape(in_U_cell_mat[-1,:],(1,3)),axis=0)
    # Reconstruction
    # First do limiting
    dU_cell_l_mat=U_cell_ghost_mat[1:-1,:]-U_cell_ghost_mat[ :-2,:]
    idx_small_tuple=np.where(np.abs(dU_cell_l_mat)<1E-12)
    idx_small_r=idx_small_tuple[0]
    idx_small_c=idx_small_tuple[1]
    dU_cell_l_mat[idx_small_r,idx_small_c]=dU_cell_l_mat[idx_small_r,idx_small_c]+(dU_cell_l_mat[idx_small_r,idx_small_c]>=0.0)*1E-12
    dU_cell_r_mat=U_cell_ghost_mat[2:  ,:]-U_cell_ghost_mat[1:-1,:]
    idx_small_tuple=np.where(np.abs(dU_cell_r_mat)<1E-12)
    idx_small_r=idx_small_tuple[0]
    idx_small_c=idx_small_tuple[1]
    dU_cell_r_mat[idx_small_r,idx_small_c]=dU_cell_r_mat[idx_small_r,idx_small_c]+(dU_cell_r_mat[idx_small_r,idx_small_c]>=0.0)*1E-12
    smooth_indic_cell_mat=dU_cell_r_mat/dU_cell_l_mat
    if(in_doLimiting):
        phi_2_cell_mat=getLimiter(1.0/smooth_indic_cell_mat,in_limiter_type)
        phi_1_cell_mat=getLimiter(smooth_indic_cell_mat,in_limiter_type)
    else:
        phi_2_cell_mat=1.0/smooth_indic_cell_mat
        phi_1_cell_mat=smooth_indic_cell_mat
    if(in_order==2):
        kappa=0.0 # linear reconstruction
    elif(in_order==3):
        kappa=1.0/3.0 # quadratic reconstruction, 3rd order
    else:
        from sys import exit
        exit("Order cannot be more than 3!")
    U_cell_lface_mat=in_U_cell_mat-0.25*(\
         (1+kappa)*(phi_2_cell_mat*dU_cell_r_mat)+(1-kappa)*(phi_1_cell_mat*dU_cell_l_mat) )
    U_cell_rface_mat=in_U_cell_mat+0.25*(\
         (1-kappa)*(phi_2_cell_mat*dU_cell_r_mat)+(1+kappa)*(phi_1_cell_mat*dU_cell_l_mat) )
    # BC: Extrapolate flux variables at both ends
    U_face_l_mat=np.append(np.reshape(U_cell_rface_mat[0,:],(1,3)),U_cell_rface_mat,axis=0)
    U_face_r_mat=np.append(U_cell_lface_mat,np.reshape(U_cell_lface_mat[-1,:],(1,3)),axis=0)
    if(in_FluxType=="ROE"):
        F_face_mat=RoeFlux(U_face_l_mat,U_face_r_mat,EntropyFix)
    elif(in_FluxType=="AUSM"):
        F_face_mat=AUSMFlux(U_face_l_mat,U_face_r_mat,in_isAUSMPlus)
    else:
        from sys import exit
        exit("Not implemented!")
    F_l_mat=F_face_mat[:-1,:]
    F_r_mat=F_face_mat[1:,:]
    dF_mat=F_r_mat-F_l_mat
    dF_mat/=in_dx
    return dF_mat

def getLimiter(in_smooth_indic_mat,in_limiter_type):
    from sys import exit
    if(in_limiter_type=="MINMOD"):
        from numpy import zeros,ones,maximum,minimum
        phi_mat=maximum(zeros(in_smooth_indic_mat.shape),minimum(in_smooth_indic_mat,ones(in_smooth_indic_mat.shape)))
    elif(in_limiter_type=="VANLEER"):
        exit("Not implemented!")
    else:
        exit("Not implemented!")
    return phi_mat

def RoeFlux(in_U_l_mat,in_U_r_mat,in_EntropyFix):
    F_l_mat=Flux(in_U_l_mat)
    F_r_mat=Flux(in_U_r_mat)
    rho_l_arr=in_U_l_mat[:,0]
    rho_r_arr=in_U_r_mat[:,0]
    u_l_arr=in_U_l_mat[:,1]/rho_l_arr
    u_r_arr=in_U_r_mat[:,1]/rho_r_arr
    p_l_arr=F_l_mat[:,1]-rho_l_arr*u_l_arr**2
    p_r_arr=F_r_mat[:,1]-rho_r_arr*u_r_arr**2
    H_l_arr=(in_U_l_mat[:,2]+p_l_arr)/rho_l_arr
    H_r_arr=(in_U_r_mat[:,2]+p_r_arr)/rho_r_arr
    R_arr=np.sqrt(rho_r_arr/rho_l_arr)
    rho_aver_arr=R_arr*rho_l_arr
    u_aver_arr=(R_arr*u_r_arr+u_l_arr)/(R_arr+1)
    H_aver_arr=(R_arr*H_r_arr+H_l_arr)/(R_arr+1)
    c_aver_arr=np.sqrt((GAMMA-1)*(H_aver_arr-0.5*u_aver_arr**2))
    lambda_1_aver_arr=u_aver_arr*np.ones(u_aver_arr.shape) # otherwise they are linked.
    lambda_2_aver_arr=u_aver_arr+c_aver_arr
    lambda_3_aver_arr=u_aver_arr-c_aver_arr
    lambda_1_aver_arr=np.abs(lambda_1_aver_arr)
    lambda_2_aver_arr=np.abs(lambda_2_aver_arr)
    lambda_3_aver_arr=np.abs(lambda_3_aver_arr)
    if in_EntropyFix==1:
        lambda_1_l_arr=u_l_arr*np.ones(u_l_arr.shape)
        lambda_1_r_arr=u_r_arr*np.ones(u_r_arr.shape)
        epsilon_1_arr=np.maximum(np.maximum(lambda_1_aver_arr-lambda_1_l_arr,lambda_1_r_arr-lambda_1_aver_arr),np.zeros(lambda_1_aver_arr.shape))
        lambda_1_aver_arr=(lambda_1_aver_arr>=epsilon_1_arr)*lambda_1_aver_arr+(lambda_1_aver_arr<epsilon_1_arr)*epsilon_1_arr
        c_l_arr=np.sqrt(GAMMA*p_l_arr/rho_l_arr)
        c_r_arr=np.sqrt(GAMMA*p_r_arr/rho_r_arr)
        lambda_2_l_arr=u_l_arr+c_l_arr
        lambda_2_r_arr=u_r_arr+c_r_arr
        epsilon_2_arr=np.maximum(np.maximum(lambda_2_aver_arr-lambda_2_l_arr,lambda_2_r_arr-lambda_2_aver_arr),np.zeros(lambda_2_aver_arr.shape))
        lambda_2_aver_arr=(lambda_2_aver_arr>=epsilon_2_arr)*lambda_2_aver_arr+(lambda_2_aver_arr<epsilon_2_arr)*epsilon_2_arr
        lambda_3_l_arr=u_l_arr-c_l_arr
        lambda_3_r_arr=u_r_arr-c_r_arr
        epsilon_3_arr=np.maximum(np.maximum(lambda_3_aver_arr-lambda_3_l_arr,lambda_3_r_arr-lambda_3_aver_arr),np.zeros(lambda_3_aver_arr.shape))
        lambda_3_aver_arr=(lambda_3_aver_arr>=epsilon_3_arr)*lambda_3_aver_arr+(lambda_3_aver_arr<epsilon_3_arr)*epsilon_3_arr
    eigvec_1_aver_mat=np.ones((lambda_1_aver_arr.size,3))
    eigvec_2_aver_mat=np.ones((lambda_2_aver_arr.size,3))
    eigvec_3_aver_mat=np.ones((lambda_3_aver_arr.size,3))
    eigvec_1_aver_mat[:,1]=u_aver_arr*np.ones(u_aver_arr.shape)
    eigvec_1_aver_mat[:,2]=0.5*u_aver_arr**2
    eigvec_2_aver_mat[:,1]=u_aver_arr+c_aver_arr
    eigvec_2_aver_mat[:,2]=H_aver_arr+u_aver_arr*c_aver_arr
    tmp_arr=0.5*rho_aver_arr/c_aver_arr
    eigvec_2_aver_mat[:,0]*=tmp_arr
    eigvec_2_aver_mat[:,1]*=tmp_arr
    eigvec_2_aver_mat[:,2]*=tmp_arr
    eigvec_3_aver_mat[:,1]=u_aver_arr-c_aver_arr
    eigvec_3_aver_mat[:,2]=H_aver_arr-u_aver_arr*c_aver_arr
    eigvec_3_aver_mat[:,0]*=-tmp_arr
    eigvec_3_aver_mat[:,1]*=-tmp_arr
    eigvec_3_aver_mat[:,2]*=-tmp_arr
    du_aver_arr=u_r_arr-u_l_arr
    dp_aver_arr=p_r_arr-p_l_arr
    drho_aver_arr=rho_r_arr-rho_l_arr
    wave_1_mag_arr=drho_aver_arr-dp_aver_arr/c_aver_arr**2
    wave_2_mag_arr=du_aver_arr+dp_aver_arr/(rho_aver_arr*c_aver_arr)
    wave_3_mag_arr=du_aver_arr-dp_aver_arr/(rho_aver_arr*c_aver_arr)
    tmp_mat=np.tile(lambda_1_aver_arr,(3,1)).T * np.tile(wave_1_mag_arr,(3,1)).T * eigvec_1_aver_mat
    tmp_mat+=np.tile(lambda_2_aver_arr,(3,1)).T * np.tile(wave_2_mag_arr,(3,1)).T * eigvec_2_aver_mat
    tmp_mat+=np.tile(lambda_3_aver_arr,(3,1)).T * np.tile(wave_3_mag_arr,(3,1)).T * eigvec_3_aver_mat
    F_mat=0.5*(F_l_mat+F_r_mat-tmp_mat)
    return F_mat

def AUSMFlux(in_U_l_mat,in_U_r_mat,in_isPlus):
    V_l_mat=U2V_mat(in_U_l_mat)
    V_r_mat=U2V_mat(in_U_r_mat)
    c_l_arr=np.sqrt(GAMMA*V_l_mat[:,2]/V_l_mat[:,0])
    c_r_arr=np.sqrt(GAMMA*V_r_mat[:,2]/V_r_mat[:,0])
    Ma_l_arr=V_l_mat[:,1]/c_l_arr
    Ma_r_arr=V_r_mat[:,1]/c_r_arr
    # P_ma
    if(in_isPlus):
        alpha=0.1875
    else:
        alpha=0.0
    P_ma_l_up_arr=np.zeros(Ma_l_arr.shape)
    idx_sup_arr=np.abs(Ma_l_arr)>=1.0
    P_ma_l_up_arr[idx_sup_arr]=0.5*(1+np.sign(Ma_l_arr[idx_sup_arr]))
    idx_sub_arr=np.abs(Ma_l_arr)<1.0
    P_ma_l_up_arr[idx_sub_arr]=0.25*(Ma_l_arr[idx_sub_arr]+1.0)**2*(2.0-Ma_l_arr[idx_sub_arr])\
        +alpha*Ma_l_arr[idx_sub_arr]*(Ma_l_arr[idx_sub_arr]**2-1.0)**2
    P_ma_r_down_arr=np.zeros(Ma_r_arr.shape)
    idx_sup_arr=np.abs(Ma_r_arr)>=1.0
    P_ma_r_down_arr[idx_sup_arr]=0.5*(1-np.sign(Ma_r_arr[idx_sup_arr]))
    idx_sub_arr=np.abs(Ma_r_arr)<1.0
    P_ma_r_down_arr[idx_sub_arr]=0.25*(Ma_r_arr[idx_sub_arr]-1.0)**2*(2.0+Ma_r_arr[idx_sub_arr])\
        -alpha*Ma_r_arr[idx_sub_arr]*(Ma_r_arr[idx_sub_arr]**2-1.0)**2
    P_arr=V_l_mat[:,2]*P_ma_l_up_arr + V_r_mat[:,2]*P_ma_r_down_arr
    P_mat=np.zeros(in_U_l_mat.shape)
    P_mat[:,1]=P_arr
    # M_ma
    if(in_isPlus):
        beta=0.125
    else:
        beta=0.0
    M_ma_l_up_arr=np.zeros(Ma_l_arr.shape)
    M_ma_r_down_arr=np.zeros(Ma_l_arr.shape)
    idx_sup_arr=np.abs(Ma_l_arr)>1.0
    M_ma_l_up_arr[idx_sup_arr]=0.5*(Ma_l_arr[idx_sup_arr]+np.abs(Ma_l_arr[idx_sup_arr]))
    idx_sub_arr=np.abs(Ma_l_arr)<=1.0
    M_ma_l_up_arr[idx_sub_arr]=0.25*(Ma_l_arr[idx_sub_arr]+1.0)**2\
        +beta*(Ma_l_arr[idx_sub_arr]**2-1.0)**2
    idx_sup_arr=np.abs(Ma_r_arr)>1.0
    M_ma_r_down_arr[idx_sup_arr]=0.5*(Ma_r_arr[idx_sup_arr]-np.abs(Ma_r_arr[idx_sup_arr]))
    idx_sub_arr=np.abs(Ma_r_arr)<=1.0
    M_ma_r_down_arr[idx_sub_arr]=-0.25*(Ma_r_arr[idx_sub_arr]-1.0)**2\
        -beta*(Ma_r_arr[idx_sub_arr]**2-1.0)**2
    M_arr=M_ma_l_up_arr+M_ma_r_down_arr
    F_l_mat=Flux(in_U_l_mat)
    F_r_mat=Flux(in_U_r_mat)
    Phi_l_mat=np.zeros(in_U_l_mat.shape)
    if(in_isPlus):
        Phi_l_mat[:,0]=V_l_mat[:,0]
        Phi_l_mat[:,1]=in_U_l_mat[:,1]
        Phi_l_mat[:,2]=(in_U_l_mat[:,2]+V_l_mat[:,2])
    else:
        Phi_l_mat[:,0]=V_l_mat[:,0]*c_l_arr
        Phi_l_mat[:,1]=in_U_l_mat[:,1]*c_l_arr
        Phi_l_mat[:,2]=(in_U_l_mat[:,2]+V_l_mat[:,2])*c_l_arr
    Phi_r_mat=np.zeros(in_U_r_mat.shape)
    if(in_isPlus):
        Phi_r_mat[:,0]=V_r_mat[:,0]
        Phi_r_mat[:,1]=in_U_r_mat[:,1]
        Phi_r_mat[:,2]=(in_U_r_mat[:,2]+V_r_mat[:,2])
    else:
        Phi_r_mat[:,0]=V_r_mat[:,0]*c_r_arr
        Phi_r_mat[:,1]=in_U_r_mat[:,1]*c_r_arr
        Phi_r_mat[:,2]=(in_U_r_mat[:,2]+V_r_mat[:,2])*c_r_arr
    Phi_mat=np.zeros(Phi_l_mat.shape)
    idx_pos_arr=M_arr>=0.0
    Phi_mat[idx_pos_arr,:]=Phi_l_mat[idx_pos_arr,:]
    idx_neg_arr=M_arr<0.0
    Phi_mat[idx_neg_arr,:]=Phi_r_mat[idx_neg_arr,:]
    if(in_isPlus):
        c_arr=np.zeros(c_l_arr.shape)
        c_arr[idx_pos_arr]=c_l_arr[idx_pos_arr]
        c_arr[idx_neg_arr]=c_r_arr[idx_neg_arr]
        Fc_mat=np.tile(np.reshape(M_arr,(M_arr.size,1)),(1,3))*np.tile(np.reshape(c_arr,(c_arr.size,1)),(1,3))*Phi_mat
    else:
        Fc_mat=np.tile(np.reshape(M_arr,(M_arr.size,1)),(1,3))*Phi_mat
    F_mat=Fc_mat+P_mat
    return F_mat

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('sjc')
    x_l_edge=-1.0
    x_h_edge=1.0
    n_x_edge=100
    n_x_edge+=1
    x_edge_arr=np.linspace(x_l_edge,x_h_edge,n_x_edge)
    x_cell_arr=(x_edge_arr[0:-1]+x_edge_arr[1:])/2.0
    IC_type=1
    case=1
    if case==1:
        # Classic SOD shock tube
        rho_l_init=1.0
        u_l_init=0.0
        p_l_init=1.0
        rho_r_init=0.125
        u_r_init=0.0
        p_r_init=0.1
    elif case==2:
        # Sonic rarefaction
        rho_l_init=3.857
        u_l_init=0.92
        p_l_init=10.333
        rho_r_init=1.0
        u_r_init=3.55
        p_r_init=1.0
    elif case==3:
        # Vacuum
        rho_l_init=1.0
        u_l_init=-0.2
        p_l_init=1.0
        rho_r_init=1.0
        u_r_init=0.2
        p_r_init=1.0
    else:
        exit("Not implemented!")
    U_cell_mat=IC(x_cell_arr,IC_type)
    dx=x_edge_arr[1]-x_edge_arr[0]
    cfl=0.9
    order=2
    #  FluxType="ROE"
    EntropyFix=1
    FluxType="AUSM"
    isAUSMPlus=0
    doLimiting=1
    limiter_type="MINMOD"
    plot_progress=False
    #  plot_progress=True
    n_iter=100000
    #  n_iter=3
    end_time=0.2
    time=0
    i_iter=0
    while(time<end_time and i_iter<n_iter):
        V_cell_mat=U2V_mat(U_cell_mat)
        if(plot_progress):
            fig=plt.figure()
            ax=fig.gca()
            ax.plot(x_cell_arr,V_cell_mat[:,0],'o-',label=r"$\rho$")
            ax.plot(x_cell_arr,V_cell_mat[:,1],'s-',label=r"$u$")
            ax.plot(x_cell_arr,V_cell_mat[:,2],'v-',label=r"$p$")
            ax.set_xlabel("X")
            #  ax.set_ylabel("U")
            #  ax.set_xlim([0.45,0.6])
            ax.legend()
            ax.set_title("it=%d,t=%.3f"%(i_iter,time))
            plt.show()
            plt.close(fig)
        i_iter+=1
        c_cell_arr=np.sqrt(GAMMA*V_cell_mat[:,2]/V_cell_mat[:,0])
        dt=cfl*dx/np.max(np.abs(V_cell_mat[:,1])+c_cell_arr)
        if(time+dt>end_time):
            dt=end_time-time
        time+=dt
        #  U_cell_mat=RK1(U_cell_mat,dt,dx)
        U_cell_mat=RK2(U_cell_mat,dt,dx,order,doLimiting,limiter_type,FluxType,isAUSMPlus)
    V_cell_mat=U2V_mat(U_cell_mat)
    data_save_mat=np.zeros((x_cell_arr.size,4))
    data_save_mat[:,0]=x_cell_arr
    data_save_mat[:,1]=V_cell_mat[:,0]
    data_save_mat[:,2]=V_cell_mat[:,1]
    data_save_mat[:,3]=V_cell_mat[:,2]
    np.savetxt("solution.dat",data_save_mat,header="x,rho,u,p")
    fig=plt.figure()
    ax=fig.gca()
    ax.plot(x_cell_arr,V_cell_mat[:,0],'o-',label=r"$\rho$")
    ax.plot(x_cell_arr,V_cell_mat[:,1],'s-',label=r"$u$")
    ax.plot(x_cell_arr,V_cell_mat[:,2],'v-',label=r"$p$")
    ax.set_xlim([-1.0,1.0])
    ax.set_ylim([-0.05,1.0])
    ax.set_xlabel("X")
    #  ax.set_ylabel("U")
    ax.legend()
    fig_name="U_it%d_t%.3f.png"%(i_iter,time)
    plt.savefig(fig_name)
