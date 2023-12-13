from mixed_pdf_tools import pdf_norm_expon_mixed
from numba_stats import norm, expon
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def plot_signal_background_mixed(
    f,
    la,
    mu,
    sg,
    alpha,
    beta,
    filename,
    npoints=1001,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    x_plot = np.linspace(alpha, beta, npoints)
    pdf_s = norm.pdf(x_plot, loc=mu, scale=sg)
    pdf_b = expon.pdf(x_plot, loc=0, scale=1 / la)
    pdf_sb = pdf_norm_expon_mixed(x_plot, f, la, mu, sg, alpha, beta)
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2))) - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))
    ax.plot(x_plot, weight_s * pdf_s, label="Signal density")
    ax.plot(x_plot, weight_b * pdf_b, label="Background density")
    ax.plot(x_plot, pdf_sb, label="Total density")
    ax.set(
        title="Plot Showing Underlying Probability Density Functions of"
        " Signal and Background Events",
        xlabel="M",
        ylabel="Density",
    )
    ax.legend()
    plt.savefig("outputs/" + filename)


plot_signal_background_mixed(
    0.1, 0.5, 5.28, 0.018, 5.0, 5.6, "part_d_plot.png"
)
