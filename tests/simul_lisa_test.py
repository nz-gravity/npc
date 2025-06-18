import numpy as np
import matplotlib.pyplot as plt
from npc.npcor.mcmc import mcmc
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MultipleLocator
import os
import h5py
from npc.utils import compute_ci as CI
from npc.utils import compute_iae as IAE

def traceplot(samples,
              var_label,
              out_path,
              dpi=300):
    plt.figure()
    plt.plot(samples, label='MCMC samples')
    plt.xlabel('Iterations')
    plt.ylabel(var_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()



def plot_lambda_boxplots_zoom(lam_mat,
                              weight_idx=1,
                              *,
                              figsize=(10, 10),
                              whis=(5, 95),
                              showmeans=True,
                              inset_size=(0.18, 0.28),
                              inset_loc="lower left",
                              ytick_step=None,
                              dpi=300,
                              savepath=None):

    lam_mat = np.asarray(lam_mat)
    K = lam_mat.shape[1]

    # main box-plot
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(lam_mat,
                    whis=whis,
                    showmeans=showmeans,
                    meanline=True,
                    patch_artist=True,
                    widths=0.6)

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels(range(1, K + 1))

    if ytick_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(ytick_step))

    axins = inset_axes(ax,
                       width=f"{inset_size[0]*100:.0f}%",
                       height=f"{inset_size[1]*100:.0f}%",
                       loc=inset_loc,
                       borderpad=1)
    #zoomed lambda box-plot
    axins.boxplot(lam_mat[:, weight_idx - 1][:, None],
                  whis=whis,
                  showmeans=showmeans,
                  meanline=True,
                  patch_artist=True,
                  widths=0.5)
    axins.yaxis.set_label_position("right")
    axins.yaxis.tick_right()
    axins.tick_params(axis="y", labelsize=8)
    axins.spines.right.set_position(("outward", 4))
    q_lo, q_hi = np.percentile(lam_mat[:, weight_idx - 1], [2.5, 97.5])
    axins.set_ylim(q_lo, q_hi)
    axins.set_xticks([])
    axins.set_title(fr"Zoom: $\lambda_{{{weight_idx}}}$", fontsize=9, pad=3)
    x_main = weight_idx
    y_med = bp["medians"][weight_idx - 1].get_ydata()[0]
    for y_ins in [1.0, 0.0]:
        fig.add_artist(ConnectionPatch(
            xyA=(x_main, y_med), coordsA=ax.transData,
            xyB=(0.5, y_ins),   coordsB=axins.transAxes,
            color="grey", lw=0.7, zorder=0))

    plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()



def plot_psd_diagnostics(f,
                         per,
                         ci_npsd,
                         myobj,
                         out_dir,
                         ci_spl=None,
                         spar=None,
                         tpsd=None,
                         dpi=300,
                         ytick_step=1e-4,
                         weight_idx=1,
                         channel="x2",
                         knot_alo=[1e-13,5e-13]):
    plt.figure()
    plt.plot(f, per, color='black', label=f'{channel} channel', alpha=0.2)
    plt.plot(f, np.exp(ci_npsd.med), color='red', label=f'Estimated {channel} noise PSD')
    plt.fill_between(f, np.exp(ci_npsd.u05), np.exp(ci_npsd.u95),
                     color='red', alpha=0.3, linewidth=0)
    if tpsd is not None:
        plt.plot(f, tpsd , color='black', linestyle='--',
             label=f'True {channel} channel noise PSD')
    if ci_spl is not None:
        plt.plot(f, np.exp(ci_spl.med), color='purple', label='Estimated spline PSD')
        plt.fill_between(f, np.exp(ci_spl.u05), np.exp(ci_spl.u95),
                     color='purple', alpha=0.3, linewidth=0)
    if spar is not None:
        plt.plot(f, spar, color='green', label='Parametric model')

    plt.vlines(myobj.knots, knot_alo[0],knot_alo[1], colors='orange', label='knots')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{out_dir}/psd_{channel}.png', dpi=dpi, bbox_inches='tight')
    plt.show()

    plot_lambda_boxplots_zoom(myobj.lambda_matrix,
                          weight_idx=weight_idx,
                          inset_loc="lower left",
                          ytick_step=ytick_step,
                          savepath=f"{out_dir}/lambda_box_zoom{weight_idx}.png")


    traceplot(samples   = myobj.loglikelihood,
              var_label = 'Log-likelihood',
              out_path  = f'{out_dir}/likelihood.png',
              dpi       = dpi)

def run_mcmc_and_summarise(
        out_dir,
        f,
        per,
        n,
        burnin,
        n_weights,
        spar=1,
        blocked=False,
        tpsd=None):
    t0 = time.time()
    myobj = mcmc(n=n,
                 f=f,
                 per=per,
                 burnin=burnin,
                 Spar=spar,
                 n_weights=n_weights,
                 blocked=blocked)

    runtime = time.time() - t0
    h5_path = os.path.join(out_dir, "mcmc_results.h5")
    ci_spl   = CI(myobj.splines_psd)
    ci_npsd = CI(myobj.noise_psd)
    iae = None
    if tpsd is not None:
        iae= IAE(np.exp(ci_npsd.med), tpsd, len(tpsd))

    with h5py.File(h5_path, "w") as h5f:
        g = h5f.create_group("mcmc")
        g.create_dataset("lam_mat",      data=myobj.lambda_matrix,   compression="gzip")
        g.create_dataset("splines_mat",  data=myobj.splines_psd, compression="gzip")
        g.create_dataset("npsdT",        data=myobj.noise_psd,     compression="gzip")
        g.create_dataset("loglikelihood",data=myobj.loglikelihood, compression="gzip")
        g.create_dataset("npsd_u05", data=ci_npsd.u05, compression="gzip")
        g.create_dataset("npsd_u95", data=ci_npsd.u95, compression="gzip")
        g.create_dataset("npsd_med", data=ci_npsd.med, compression="gzip")
        # store scalar/meta information as attributes
        g.attrs["runtime_sec"] = runtime
        g.attrs["n"]           = myobj.n
        g.attrs["n_weights"]   = myobj.n_weights
        g.attrs["burnin"]      = myobj.burnin
        g.attrs["blocked"]     = myobj.blocked
        if tpsd is not None:
            g.attrs["iae"] = iae

    return myobj, ci_spl, ci_npsd, iae

def datainput(dire,param=True):
    T_per = np.loadtxt(f'{dire}/T_per.txt')
    T = np.loadtxt(f'{dire}/T.txt')
    f = np.loadtxt(f'{dire}/f.txt')
    std_T = np.loadtxt(f'{dire}/std_T.txt')
    spar=1
    if param:
        spar=np.loadtxt(f'{dire}/spar.txt')
    return T_per, T, f, std_T, spar

def compute_iae(psd, truepsd,n):
    return sum(abs(psd - truepsd)) * 2* np.pi / n


if True:
    np.random.seed(10)
    ndays=20
    dire = f'/home/naim769/oneMonth/pcode/first/LISA/blocked/Guilliume/{ndays}days'
    T_per, T, f, std_T, spar= datainput(dire)

    myobj,ci_spl, ci_npsdT,iae=run_mcmc_and_summarise(
        out_dir=dire,
        f=f,
        per=T_per,
        spar=spar/(std_T) ** 2,
        n=10000,
        burnin=5000,
        n_weights=15,
        blocked=True,
        tpsd=T)

    plot_psd_diagnostics(f=f,
                         per=T_per[0],
                         ci_npsd=ci_npsdT,
                         myobj=myobj,
                         out_dir= dire,
                         ci_spl=ci_spl,
                         spar=spar/(std_T) ** 2,
                         tpsd= T/(std_T) ** 2,
                         dpi=300,
                         ytick_step=1e-3,
                         weight_idx=1,
                         channel="T",
                         knot_alo=[1e-5,5e-5])
