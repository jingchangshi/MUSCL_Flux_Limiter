import numpy as np

fdir="../2nd_order_minmod/"

data=np.loadtxt(fdir+"RoeEntrFix_cfl0.1.dat")
order2_roe1_entrfix_xc_arr=data[:,0]
order2_roe1_entrfix_rho_arr=data[:,1]
order2_roe1_entrfix_u_arr=data[:,2]
order2_roe1_entrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeEntrFix_cfl0.3.dat")
order2_roe3_entrfix_xc_arr=data[:,0]
order2_roe3_entrfix_rho_arr=data[:,1]
order2_roe3_entrfix_u_arr=data[:,2]
order2_roe3_entrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeEntrFix_cfl0.9.dat")
order2_roe9_entrfix_xc_arr=data[:,0]
order2_roe9_entrfix_rho_arr=data[:,1]
order2_roe9_entrfix_u_arr=data[:,2]
order2_roe9_entrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeNoEntrFix_cfl0.1.dat")
order2_roe1_noentrfix_xc_arr=data[:,0]
order2_roe1_noentrfix_rho_arr=data[:,1]
order2_roe1_noentrfix_u_arr=data[:,2]
order2_roe1_noentrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeNoEntrFix_cfl0.3.dat")
order2_roe3_noentrfix_xc_arr=data[:,0]
order2_roe3_noentrfix_rho_arr=data[:,1]
order2_roe3_noentrfix_u_arr=data[:,2]
order2_roe3_noentrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeNoEntrFix_cfl0.9.dat")
order2_roe9_noentrfix_xc_arr=data[:,0]
order2_roe9_noentrfix_rho_arr=data[:,1]
order2_roe9_noentrfix_u_arr=data[:,2]
order2_roe9_noentrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"AUSM_cfl0.1.dat")
order2_ausm1_minmod_xc_arr=data[:,0]
order2_ausm1_minmod_rho_arr=data[:,1]
order2_ausm1_minmod_u_arr=data[:,2]
order2_ausm1_minmod_p_arr=data[:,3]

data=np.loadtxt(fdir+"AUSM_cfl0.3.dat")
order2_ausm3_minmod_xc_arr=data[:,0]
order2_ausm3_minmod_rho_arr=data[:,1]
order2_ausm3_minmod_u_arr=data[:,2]
order2_ausm3_minmod_p_arr=data[:,3]

fdir="../3rd_order_minmod/"

data=np.loadtxt(fdir+"RoeEntrFix_cfl0.1.dat")
order3_roe1_entrfix_xc_arr=data[:,0]
order3_roe1_entrfix_rho_arr=data[:,1]
order3_roe1_entrfix_u_arr=data[:,2]
order3_roe1_entrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeEntrFix_cfl0.3.dat")
order3_roe3_entrfix_xc_arr=data[:,0]
order3_roe3_entrfix_rho_arr=data[:,1]
order3_roe3_entrfix_u_arr=data[:,2]
order3_roe3_entrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeEntrFix_cfl0.9.dat")
order3_roe9_entrfix_xc_arr=data[:,0]
order3_roe9_entrfix_rho_arr=data[:,1]
order3_roe9_entrfix_u_arr=data[:,2]
order3_roe9_entrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeNoEntrFix_cfl0.1.dat")
order3_roe1_noentrfix_xc_arr=data[:,0]
order3_roe1_noentrfix_rho_arr=data[:,1]
order3_roe1_noentrfix_u_arr=data[:,2]
order3_roe1_noentrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeNoEntrFix_cfl0.3.dat")
order3_roe3_noentrfix_xc_arr=data[:,0]
order3_roe3_noentrfix_rho_arr=data[:,1]
order3_roe3_noentrfix_u_arr=data[:,2]
order3_roe3_noentrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"RoeNoEntrFix_cfl0.9.dat")
order3_roe9_noentrfix_xc_arr=data[:,0]
order3_roe9_noentrfix_rho_arr=data[:,1]
order3_roe9_noentrfix_u_arr=data[:,2]
order3_roe9_noentrfix_p_arr=data[:,3]

data=np.loadtxt(fdir+"AUSM_cfl0.1.dat")
order3_ausm1_minmod_xc_arr=data[:,0]
order3_ausm1_minmod_rho_arr=data[:,1]
order3_ausm1_minmod_u_arr=data[:,2]
order3_ausm1_minmod_p_arr=data[:,3]

data=np.loadtxt(fdir+"AUSM_cfl0.3.dat")
order3_ausm3_minmod_xc_arr=data[:,0]
order3_ausm3_minmod_rho_arr=data[:,1]
order3_ausm3_minmod_u_arr=data[:,2]
order3_ausm3_minmod_p_arr=data[:,3]

fdir="../2nd_order_nolimiter/"

data=np.loadtxt(fdir+"AUSM_cfl0.1.dat")
order2_ausm1_nolim_xc_arr=data[:,0]
order2_ausm1_nolim_rho_arr=data[:,1]
order2_ausm1_nolim_u_arr=data[:,2]
order2_ausm1_nolim_p_arr=data[:,3]

data=np.loadtxt(fdir+"AUSM_cfl0.3.dat")
order2_ausm3_nolim_xc_arr=data[:,0]
order2_ausm3_nolim_rho_arr=data[:,1]
order2_ausm3_nolim_u_arr=data[:,2]
order2_ausm3_nolim_p_arr=data[:,3]

fdir="../3rd_order_nolimiter/"

data=np.loadtxt(fdir+"AUSM_cfl0.1.dat")
order3_ausm1_nolim_xc_arr=data[:,0]
order3_ausm1_nolim_rho_arr=data[:,1]
order3_ausm1_nolim_u_arr=data[:,2]
order3_ausm1_nolim_p_arr=data[:,3]

data=np.loadtxt(fdir+"AUSM_cfl0.3.dat")
order3_ausm3_nolim_xc_arr=data[:,0]
order3_ausm3_nolim_rho_arr=data[:,1]
order3_ausm3_nolim_u_arr=data[:,2]
order3_ausm3_nolim_p_arr=data[:,3]

import matplotlib.pyplot as plt
plt.style.use('sjc')

roe1_noentrfix_label="Roe,NoEntropyFix,CFL=0.1"
roe3_noentrfix_label="Roe,NoEntropyFix,CFL=0.3"
roe9_noentrfix_label="Roe,NoEntropyFix,CFL=0.9"
roe1_entrfix_label="Roe,EntropyFix,CFL=0.1"
roe3_entrfix_label="Roe,EntropyFix,CFL=0.3"
roe9_entrfix_label="Roe,EntropyFix,CFL=0.9"
ausm1_minmod_label="AUSM,CFL=0.1,MINMOD"
ausm3_minmod_label="AUSM,CFL=0.3,MINMOD"
ausm1_nolim_label="AUSM,CFL=0.1,NoLimiter"
ausm3_nolim_label="AUSM,CFL=0.3,NoLimiter"

order2_label=r"linear: $\kappa=0.0$"
order3_label=r"quadratic: $\kappa=1/3$"

ds_step=4
ds_longshort_1=[ds_step*2,ds_step,ds_step*4,ds_step]
ds_step=2
ds_longshort_2=[ds_step*2,ds_step,ds_step*4,ds_step]
ds_step=4
ds_shortlong_1=[ds_step,ds_step*2,ds_step,ds_step*2]
ds_step=2
ds_shortlong_2=[ds_step,ds_step*2,ds_step,ds_step*2]

xlim=[-1.0,1.0]
ylim=[-0.05,1.0]

# Base on Roe-NoEntropyFix
fig=plt.figure()
ax=fig.gca()
ax.plot(order2_roe1_noentrfix_xc_arr,order2_roe1_noentrfix_rho_arr,'-',label=order2_label+r", $\rho$")
ax.plot(order3_roe1_noentrfix_xc_arr,order3_roe1_noentrfix_rho_arr,':',label=order3_label+r", $\rho$")
ax.plot(order2_roe1_noentrfix_xc_arr,order2_roe1_noentrfix_u_arr,'--',dashes=ds_longshort_1,label=order2_label+r", $u$")
ax.plot(order3_roe1_noentrfix_xc_arr,order3_roe1_noentrfix_u_arr,'--',dashes=ds_shortlong_1,label=order3_label+r", $u$")
ax.plot(order2_roe1_noentrfix_xc_arr,order2_roe1_noentrfix_p_arr,'--',dashes=ds_longshort_2,label=order2_label+r", $p$")
ax.plot(order3_roe1_noentrfix_xc_arr,order3_roe1_noentrfix_p_arr,'--',dashes=ds_shortlong_2,label=order3_label+r", $p$")
ax.set_xlabel("X")
#  ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig(roe1_noentrfix_label+".png")

# Base on Roe-EntropyFix
fig=plt.figure()
ax=fig.gca()
ax.plot(order2_roe1_entrfix_xc_arr,order2_roe1_entrfix_rho_arr,'-',label=order2_label+r", $\rho$")
ax.plot(order3_roe1_entrfix_xc_arr,order3_roe1_entrfix_rho_arr,':',label=order3_label+r", $\rho$")
ax.plot(order2_roe1_entrfix_xc_arr,order2_roe1_entrfix_u_arr,'--',dashes=ds_longshort_1,label=order2_label+r", $u$")
ax.plot(order3_roe1_entrfix_xc_arr,order3_roe1_entrfix_u_arr,'--',dashes=ds_shortlong_1,label=order3_label+r", $u$")
ax.plot(order2_roe1_entrfix_xc_arr,order2_roe1_entrfix_p_arr,'--',dashes=ds_longshort_2,label=order2_label+r", $p$")
ax.plot(order3_roe1_entrfix_xc_arr,order3_roe1_entrfix_p_arr,'--',dashes=ds_shortlong_2,label=order3_label+r", $p$")
ax.set_xlabel("X")
#  ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig(roe1_entrfix_label+".png")

# Base on AUSM
fig=plt.figure()
ax=fig.gca()
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_rho_arr,'-',label=order2_label+r", $\rho$")
ax.plot(order3_ausm1_minmod_xc_arr,order3_ausm1_minmod_rho_arr,':',label=order3_label+r", $\rho$")
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_u_arr,'--',dashes=ds_longshort_1,label=order2_label+r", $u$")
ax.plot(order3_ausm1_minmod_xc_arr,order3_ausm1_minmod_u_arr,'--',dashes=ds_shortlong_1,label=order3_label+r", $u$")
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_p_arr,'--',dashes=ds_longshort_2,label=order2_label+r", $p$")
ax.plot(order3_ausm1_minmod_xc_arr,order3_ausm1_minmod_p_arr,'--',dashes=ds_shortlong_2,label=order3_label+r", $p$")
ax.set_xlabel("X")
#  ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig(ausm1_minmod_label+".png")

# Base on AUSM, No limiter
fig=plt.figure()
ax=fig.gca()
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_rho_arr,'-',label=order2_label+r", $\rho$")
ax.plot(order3_ausm1_nolim_xc_arr,order3_ausm1_nolim_rho_arr,':',label=order3_label+r", $\rho$")
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_u_arr,'--',dashes=ds_longshort_1,label=order2_label+r", $u$")
ax.plot(order3_ausm1_nolim_xc_arr,order3_ausm1_nolim_u_arr,'--',dashes=ds_shortlong_1,label=order3_label+r", $u$")
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_p_arr,'--',dashes=ds_longshort_2,label=order2_label+r", $p$")
ax.plot(order3_ausm1_nolim_xc_arr,order3_ausm1_nolim_p_arr,'--',dashes=ds_shortlong_2,label=order3_label+r", $p$")
ax.set_xlabel("X")
#  ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig(ausm1_nolim_label+".png")

