import numpy as np

data=np.loadtxt("RoeEntrFix_cfl0.1.dat")
roe1_entrfix_xc_arr=data[:,0]
roe1_entrfix_rho_arr=data[:,1]
roe1_entrfix_u_arr=data[:,2]
roe1_entrfix_p_arr=data[:,3]

data=np.loadtxt("RoeEntrFix_cfl0.3.dat")
roe3_entrfix_xc_arr=data[:,0]
roe3_entrfix_rho_arr=data[:,1]
roe3_entrfix_u_arr=data[:,2]
roe3_entrfix_p_arr=data[:,3]

data=np.loadtxt("RoeEntrFix_cfl0.9.dat")
roe9_entrfix_xc_arr=data[:,0]
roe9_entrfix_rho_arr=data[:,1]
roe9_entrfix_u_arr=data[:,2]
roe9_entrfix_p_arr=data[:,3]

data=np.loadtxt("RoeNoEntrFix_cfl0.1.dat")
roe1_noentrfix_xc_arr=data[:,0]
roe1_noentrfix_rho_arr=data[:,1]
roe1_noentrfix_u_arr=data[:,2]
roe1_noentrfix_p_arr=data[:,3]

data=np.loadtxt("RoeNoEntrFix_cfl0.3.dat")
roe3_noentrfix_xc_arr=data[:,0]
roe3_noentrfix_rho_arr=data[:,1]
roe3_noentrfix_u_arr=data[:,2]
roe3_noentrfix_p_arr=data[:,3]

data=np.loadtxt("RoeNoEntrFix_cfl0.9.dat")
roe9_noentrfix_xc_arr=data[:,0]
roe9_noentrfix_rho_arr=data[:,1]
roe9_noentrfix_u_arr=data[:,2]
roe9_noentrfix_p_arr=data[:,3]

data=np.loadtxt("AUSM_cfl0.1.dat")
ausm1_xc_arr=data[:,0]
ausm1_rho_arr=data[:,1]
ausm1_u_arr=data[:,2]
ausm1_p_arr=data[:,3]

data=np.loadtxt("AUSM_cfl0.3.dat")
ausm3_xc_arr=data[:,0]
ausm3_rho_arr=data[:,1]
ausm3_u_arr=data[:,2]
ausm3_p_arr=data[:,3]

import matplotlib.pyplot as plt
plt.style.use('sjc')

roe1_noentrfix_label="Roe,NoEntropyFix,CFL=0.1"
roe3_noentrfix_label="Roe,NoEntropyFix,CFL=0.3"
roe9_noentrfix_label="Roe,NoEntropyFix,CFL=0.9"
roe1_entrfix_label="Roe,EntropyFix,CFL=0.1"
roe3_entrfix_label="Roe,EntropyFix,CFL=0.3"
roe9_entrfix_label="Roe,EntropyFix,CFL=0.9"
ausm1_label="AUSM,CFL=0.1"
ausm3_label="AUSM,CFL=0.3"

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

# Compare Roe-NoEntropyFix against Roe-EntropyFix
fig=plt.figure()
ax=fig.gca()
ax.plot(roe1_noentrfix_xc_arr,roe1_noentrfix_rho_arr,'-',label=roe1_noentrfix_label)
ax.plot(roe3_noentrfix_xc_arr,roe3_noentrfix_rho_arr,'--',dashes=ds_longshort_1,label=roe3_noentrfix_label)
ax.plot(roe9_noentrfix_xc_arr,roe9_noentrfix_rho_arr,'--',dashes=ds_longshort_2,label=roe9_noentrfix_label)
ax.plot(roe1_entrfix_xc_arr,roe1_entrfix_rho_arr,':',label=roe1_entrfix_label)
ax.plot(roe3_entrfix_xc_arr,roe3_entrfix_rho_arr,'--',dashes=ds_shortlong_1,label=roe3_entrfix_label)
ax.plot(roe9_entrfix_xc_arr,roe9_entrfix_rho_arr,'--',dashes=ds_shortlong_2,label=roe9_entrfix_label)
ax.set_xlabel("X")
ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("RoeEntropyFix_rho.png")

fig=plt.figure()
ax=fig.gca()
ax.plot(roe1_noentrfix_xc_arr,roe1_noentrfix_u_arr,'-',label=roe1_noentrfix_label)
ax.plot(roe3_noentrfix_xc_arr,roe3_noentrfix_u_arr,'--',dashes=ds_longshort_1,label=roe3_noentrfix_label)
ax.plot(roe9_noentrfix_xc_arr,roe9_noentrfix_u_arr,'--',dashes=ds_longshort_2,label=roe9_noentrfix_label)
ax.plot(roe1_entrfix_xc_arr,roe1_entrfix_u_arr,':',label=roe1_entrfix_label)
ax.plot(roe3_entrfix_xc_arr,roe3_entrfix_u_arr,'--',dashes=ds_shortlong_1,label=roe3_entrfix_label)
ax.plot(roe9_entrfix_xc_arr,roe9_entrfix_u_arr,'--',dashes=ds_shortlong_2,label=roe9_entrfix_label)
ax.set_xlabel("X")
ax.set_ylabel(r"$u$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("RoeEntropyFix_u.png")

fig=plt.figure()
ax=fig.gca()
ax.plot(roe1_noentrfix_xc_arr,roe1_noentrfix_p_arr,'-',label=roe1_noentrfix_label)
ax.plot(roe3_noentrfix_xc_arr,roe3_noentrfix_p_arr,'--',dashes=ds_longshort_1,label=roe3_noentrfix_label)
ax.plot(roe9_noentrfix_xc_arr,roe9_noentrfix_p_arr,'--',dashes=ds_longshort_2,label=roe9_noentrfix_label)
ax.plot(roe1_entrfix_xc_arr,roe1_entrfix_p_arr,':',label=roe1_entrfix_label)
ax.plot(roe3_entrfix_xc_arr,roe3_entrfix_p_arr,'--',dashes=ds_shortlong_1,label=roe3_entrfix_label)
ax.plot(roe9_entrfix_xc_arr,roe9_entrfix_p_arr,'--',dashes=ds_shortlong_2,label=roe9_entrfix_label)
ax.set_xlabel("X")
ax.set_ylabel(r"$p$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("RoeEntropyFix_p.png")

# Compare AUSM against Roe-EntropyFix
fig=plt.figure()
ax=fig.gca()
ax.plot(ausm1_xc_arr,ausm1_rho_arr,'-',label=ausm1_label)
ax.plot(ausm3_xc_arr,ausm3_rho_arr,'--',dashes=ds_longshort_1,label=ausm3_label)
ax.plot(roe1_entrfix_xc_arr,roe1_entrfix_rho_arr,':',label=roe1_entrfix_label)
ax.plot(roe3_entrfix_xc_arr,roe3_entrfix_rho_arr,'--',dashes=ds_shortlong_1,label=roe3_entrfix_label)
ax.plot(roe9_entrfix_xc_arr,roe9_entrfix_rho_arr,'--',dashes=ds_shortlong_2,label=roe9_entrfix_label)
ax.set_xlabel("X")
ax.set_ylabel(r"$\rho$")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("AUSMRoe_rho.png")

fig=plt.figure()
ax=fig.gca()
ax.plot(ausm1_xc_arr,ausm1_u_arr,'-',label=ausm1_label)
ax.plot(ausm3_xc_arr,ausm3_u_arr,'--',dashes=ds_longshort_1,label=ausm3_label)
ax.plot(roe1_entrfix_xc_arr,roe1_entrfix_u_arr,':',label=roe1_entrfix_label)
ax.plot(roe3_entrfix_xc_arr,roe3_entrfix_u_arr,'--',dashes=ds_shortlong_1,label=roe3_entrfix_label)
ax.plot(roe9_entrfix_xc_arr,roe9_entrfix_u_arr,'--',dashes=ds_shortlong_2,label=roe9_entrfix_label)
ax.set_xlabel("X")
ax.set_ylabel("u")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("AUSMRoe_u.png")

fig=plt.figure()
ax=fig.gca()
ax.plot(ausm1_xc_arr,ausm1_p_arr,'-',label=ausm1_label)
ax.plot(ausm3_xc_arr,ausm3_p_arr,'--',dashes=ds_longshort_1,label=ausm3_label)
ax.plot(roe1_entrfix_xc_arr,roe1_entrfix_p_arr,':',label=roe1_entrfix_label)
ax.plot(roe3_entrfix_xc_arr,roe3_entrfix_p_arr,'--',dashes=ds_shortlong_1,label=roe3_entrfix_label)
ax.plot(roe9_entrfix_xc_arr,roe9_entrfix_p_arr,'--',dashes=ds_shortlong_2,label=roe9_entrfix_label)
ax.set_xlabel("X")
ax.set_ylabel("p")
ax.legend()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.savefig("AUSMRoe_p.png")

