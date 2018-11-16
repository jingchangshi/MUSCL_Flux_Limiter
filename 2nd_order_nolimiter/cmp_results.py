import numpy as np

fdir="../2nd_order_minmod/"

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

data=np.loadtxt(fdir+"AUSM_cfl0.9.dat")
order2_ausm9_minmod_xc_arr=data[:,0]
order2_ausm9_minmod_rho_arr=data[:,1]
order2_ausm9_minmod_u_arr=data[:,2]
order2_ausm9_minmod_p_arr=data[:,3]

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

data=np.loadtxt(fdir+"AUSM_cfl0.9.dat")
order2_ausm9_nolim_xc_arr=data[:,0]
order2_ausm9_nolim_rho_arr=data[:,1]
order2_ausm9_nolim_u_arr=data[:,2]
order2_ausm9_nolim_p_arr=data[:,3]

import matplotlib.pyplot as plt
plt.style.use('sjc')

ausm1_minmod_label="AUSM,CFL=0.1,MINMOD"
ausm3_minmod_label="AUSM,CFL=0.3,MINMOD"
ausm9_minmod_label="AUSM,CFL=0.9,MINMOD"
ausm1_nolim_label="AUSM,CFL=0.1,NoLimiter"
ausm3_nolim_label="AUSM,CFL=0.3,NoLimiter"
ausm9_nolim_label="AUSM,CFL=0.9,NoLimiter"

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

fig=plt.figure()
ax=fig.gca()
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_rho_arr,'-',label=ausm1_minmod_label+r", $\rho$")
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_rho_arr,':',label=ausm1_nolim_label+r", $\rho$")
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_u_arr,'--',dashes=ds_longshort_1,label=ausm1_minmod_label+r", $u$")
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_u_arr,'--',dashes=ds_shortlong_1,label=ausm1_nolim_label+r", $u$")
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_p_arr,'--',dashes=ds_longshort_2,label=ausm1_minmod_label+r", $p$")
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_p_arr,'--',dashes=ds_shortlong_2,label=ausm1_nolim_label+r", $p$")
ax.set_xlabel("X")
#  ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("2nd_order_cfl0.1"+".png")

fig=plt.figure()
ax=fig.gca()
ax.plot(order2_ausm3_minmod_xc_arr,order2_ausm3_minmod_rho_arr,'-',label=ausm3_minmod_label+r", $\rho$")
ax.plot(order2_ausm3_nolim_xc_arr,order2_ausm3_nolim_rho_arr,':',label=ausm3_nolim_label+r", $\rho$")
ax.plot(order2_ausm3_minmod_xc_arr,order2_ausm3_minmod_u_arr,'--',dashes=ds_longshort_1,label=ausm3_minmod_label+r", $u$")
ax.plot(order2_ausm3_nolim_xc_arr,order2_ausm3_nolim_u_arr,'--',dashes=ds_shortlong_1,label=ausm3_nolim_label+r", $u$")
ax.plot(order2_ausm3_minmod_xc_arr,order2_ausm3_minmod_p_arr,'--',dashes=ds_longshort_2,label=ausm3_minmod_label+r", $p$")
ax.plot(order2_ausm3_nolim_xc_arr,order2_ausm3_nolim_p_arr,'--',dashes=ds_shortlong_2,label=ausm3_nolim_label+r", $p$")
ax.set_xlabel("X")
#  ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("2nd_order_cfl0.3"+".png")

fig=plt.figure()
ax=fig.gca()
ax.plot(order2_ausm9_minmod_xc_arr,order2_ausm9_minmod_rho_arr,'-',label=ausm9_minmod_label+r", $\rho$")
ax.plot(order2_ausm9_nolim_xc_arr,order2_ausm9_nolim_rho_arr,':',label=ausm9_nolim_label+r", $\rho$")
ax.plot(order2_ausm9_minmod_xc_arr,order2_ausm9_minmod_u_arr,'--',dashes=ds_longshort_1,label=ausm9_minmod_label+r", $u$")
ax.plot(order2_ausm9_nolim_xc_arr,order2_ausm9_nolim_u_arr,'--',dashes=ds_shortlong_1,label=ausm9_nolim_label+r", $u$")
ax.plot(order2_ausm9_minmod_xc_arr,order2_ausm9_minmod_p_arr,'--',dashes=ds_longshort_2,label=ausm9_minmod_label+r", $p$")
ax.plot(order2_ausm9_nolim_xc_arr,order2_ausm9_nolim_p_arr,'--',dashes=ds_shortlong_2,label=ausm9_nolim_label+r", $p$")
ax.set_xlabel("X")
#  ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("2nd_order_cfl0.9"+".png")

fig=plt.figure()
ax=fig.gca()
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_rho_arr,'-',label=ausm1_minmod_label)
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_rho_arr,':',label=ausm1_nolim_label)
ax.plot(order2_ausm3_minmod_xc_arr,order2_ausm3_minmod_rho_arr,'--',dashes=ds_longshort_1,label=ausm3_minmod_label)
ax.plot(order2_ausm3_nolim_xc_arr,order2_ausm3_nolim_rho_arr,'--',dashes=ds_shortlong_1,label=ausm3_nolim_label)
ax.plot(order2_ausm9_minmod_xc_arr,order2_ausm9_minmod_rho_arr,'--',dashes=ds_longshort_2,label=ausm9_minmod_label)
ax.plot(order2_ausm9_nolim_xc_arr,order2_ausm9_nolim_rho_arr,'--',dashes=ds_shortlong_2,label=ausm9_nolim_label)
ax.set_xlabel("X")
ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("rho.png")

fig=plt.figure()
ax=fig.gca()
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_u_arr,'-',label=ausm1_minmod_label)
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_u_arr,':',label=ausm1_nolim_label)
ax.plot(order2_ausm3_minmod_xc_arr,order2_ausm3_minmod_u_arr,'--',dashes=ds_longshort_1,label=ausm3_minmod_label)
ax.plot(order2_ausm3_nolim_xc_arr,order2_ausm3_nolim_u_arr,'--',dashes=ds_shortlong_1,label=ausm3_nolim_label)
ax.plot(order2_ausm9_minmod_xc_arr,order2_ausm9_minmod_u_arr,'--',dashes=ds_longshort_2,label=ausm9_minmod_label)
ax.plot(order2_ausm9_nolim_xc_arr,order2_ausm9_nolim_u_arr,'--',dashes=ds_shortlong_2,label=ausm9_nolim_label)
ax.set_xlabel("X")
ax.set_ylabel(r"$u$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("u.png")

fig=plt.figure()
ax=fig.gca()
ax.plot(order2_ausm1_minmod_xc_arr,order2_ausm1_minmod_p_arr,'-',label=ausm1_minmod_label)
ax.plot(order2_ausm1_nolim_xc_arr,order2_ausm1_nolim_p_arr,':',label=ausm1_nolim_label)
ax.plot(order2_ausm3_minmod_xc_arr,order2_ausm3_minmod_p_arr,'--',dashes=ds_longshort_1,label=ausm3_minmod_label)
ax.plot(order2_ausm3_nolim_xc_arr,order2_ausm3_nolim_p_arr,'--',dashes=ds_shortlong_1,label=ausm3_nolim_label)
ax.plot(order2_ausm9_minmod_xc_arr,order2_ausm9_minmod_p_arr,'--',dashes=ds_longshort_2,label=ausm9_minmod_label)
ax.plot(order2_ausm9_nolim_xc_arr,order2_ausm9_nolim_p_arr,'--',dashes=ds_shortlong_2,label=ausm9_nolim_label)
ax.set_xlabel("X")
ax.set_ylabel(r"$p$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("p.png")

