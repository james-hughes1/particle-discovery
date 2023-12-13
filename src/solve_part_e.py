import numpy as np
from iminuit import Minuit
from iminuit.cost import BinnedNLL
from numba_stats import norm, expon
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy.special import erf
from scipy.stats import chi2

from mixed_pdf_tools import (
    sample_sb_fast,
    pdf_norm_expon_mixed,
    cdf_norm_expon_mixed
)

# Sample data from true distribution and separate into bins.
sample_array = sample_sb_fast(100000, seed=1)
nh, xe = np.histogram(sample_array, bins=100, range=(5, 5.6))

# Perform binned ML fit of parameters.
nll = BinnedNLL(nh, xe, cdf_norm_expon_mixed)
mi = Minuit(nll, f=0.5, la=1.0, mu=5.3, sg=0.02, alpha=5.0, beta=5.6)
mi.fixed["alpha"] = True
mi.fixed["beta"] = True
mi.limits["f"] = (0, 1)
mi.limits["la"] = (1e-9, 1e2)
mi.limits["mu"] = (5.0, 5.6)
mi.limits["sg"] = (1e-9, 1e2)

# Write the outputs of the parameter estimation to a file.
with open("outputs/part_e_parameter_estimation.txt", "w") as f:
    output_str = mi.migrad().__str__()
    f.write(output_str)


# Plot the sample bins with the fit density.
def plot_sample_fit(
    f,
    la,
    mu,
    sg,
    alpha,
    beta,
    filename,
    nh,
    xe,
    npoints=1001,
):
    plt.figure(figsize=(12, 10))
    grid = gs.GridSpec(
        2, 2, height_ratios=[3, 1], hspace=0,
        width_ratios=[6, 1], wspace=0
    )
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[1, 1])
    ax4 = plt.subplot(grid[0, 1])
    ax4.set_visible(False)
    plt.axis("off")
    x_plot = np.linspace(alpha, beta, npoints)
    cx = 0.5 * (xe[1:] + xe[:-1])
    errors = nh**0.5
    ax1.errorbar(cx, nh, errors, c="k", marker="o", linestyle="")
    scale_factor = np.sum(nh) * (xe[1] - xe[0])
    pdf_s = norm.pdf(x_plot, loc=mu, scale=sg)
    pdf_b = expon.pdf(x_plot, loc=0, scale=1 / la)
    pdf_sb = pdf_norm_expon_mixed(x_plot, f, la, mu, sg, alpha, beta)
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2)))
        - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))

    ax1.plot(
        x_plot, weight_s * scale_factor * pdf_s, label="Signal density fit"
    )
    ax1.plot(
        x_plot, weight_b * scale_factor * pdf_b, label="Background density fit"
    )
    ax1.plot(
        x_plot, scale_factor * pdf_sb, label="Total density fit"
    )
    ax1.set(
        ylabel="Events", title="Plot Showing Sample with Fitted Densities and"
        " Pull Plot"
    )
    ax1.legend()
    residuals = nh - (scale_factor * pdf_norm_expon_mixed(cx, *mi.values))
    pulls = residuals / errors
    ax2.errorbar(cx, pulls, 1, c="k", marker="o", linestyle="")
    ax2.hlines(0, 5.0, 5.6, colors="green")
    ax2.set(
        xlabel="M",
        ylabel="Pull"
    )
    pull_range = (np.min(pulls) - 1, np.max(pulls) + 1)
    ax3.hist(
        pulls, bins=10, range=pull_range, density=True, alpha=0.5,
        orientation="horizontal"
    )
    x_pull_plot = np.linspace(*pull_range, 100)
    ax3.plot(
        norm.pdf(x_pull_plot, loc=0, scale=1), x_pull_plot, c="green",
        alpha=0.5
    )
    ax3.set_yticks([])
    chi2_score = np.sum(pulls**2)
    dof = len(xe) - 5
    pval = 1 - chi2.cdf(chi2_score, dof)
    ax1.text(
        x=5.02, y=1800, s="Chi sq./d.o.f = {:.3f}".format(chi2_score/dof),
        fontsize=18
    )
    ax1.text(x=5.02, y=1600, s="P-value = {:.3f}".format(pval), fontsize=18)
    plt.savefig("outputs/" + filename)


plot_sample_fit(
    *mi.values, "part_e_plot.png", nh, xe
)
