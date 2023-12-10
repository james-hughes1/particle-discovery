import numpy as np
from numba_stats import norm, expon, truncnorm, truncexpon
from scipy.special import erf
from scipy.integrate import quad
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import BinnedNLL, UnbinnedNLL
from scipy.stats import chi2


def pdf_norm_expon_mixed(x, f, la, mu, sg, alpha, beta):
    pdf_s = norm.pdf(x, loc=mu, scale=sg)
    pdf_b = expon.pdf(x, loc=0, scale=1 / la)
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2)))
        - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))
    return (weight_s * pdf_s) + (weight_b * pdf_b)


def check_normalisation(pdf, lower, upper):
    integral_value, abserr = quad(pdf, lower, upper)
    return integral_value


def cdf_norm_expon_mixed(x, f, la, mu, sg, alpha, beta):
    cdf_s = norm.cdf(x, loc=mu, scale=sg) - norm.cdf(
        alpha * np.ones(x.shape), loc=mu, scale=sg
    )
    cdf_b = expon.cdf(x, loc=0, scale=1 / la) - expon.cdf(
        alpha * np.ones(x.shape), loc=0, scale=1 / la
    )
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2)))
        - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))
    return (weight_s * cdf_s) + (weight_b * cdf_b)


def plot_signal_background_mixed(
    f,
    la,
    mu,
    sg,
    alpha,
    beta,
    filename,
    npoints=1001,
    sample_hist=None,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    x_plot = np.linspace(alpha, beta, npoints)
    if sample_hist is not None:
        nh, xe = sample_hist[0], sample_hist[1]
        cx = 0.5 * (xe[1:] + xe[:-1])
        ax.errorbar(cx, nh, nh**0.5, fmt="ko")
        scale_factor = np.sum(nh) * (xe[1] - xe[0])
    else:
        scale_factor = 1
    pdf_s = norm.pdf(x_plot, loc=mu, scale=sg)
    pdf_b = expon.pdf(x_plot, loc=0, scale=1 / la)
    pdf_sb = pdf_norm_expon_mixed(x_plot, f, la, mu, sg, alpha, beta)
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2)))
        - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))
    ax.plot(x_plot, weight_s * scale_factor * pdf_s)
    ax.plot(x_plot, weight_b * scale_factor * pdf_b)
    ax.plot(x_plot, scale_factor * pdf_sb)
    plt.savefig("outputs/" + filename)


def compute_llr(sample_array, h0_bool):
    # Separate the sample into bins.
    N = len(sample_array)
    nh, xe = np.histogram(sample_array, bins=int(N / 10), range=(5, 5.6))

    # Perform binned ML fit of parameters using signal+background model.
    nll_sb = BinnedNLL(nh, xe, cdf_h1_fast)
    mi = Minuit(nll_sb, f=0.05, la=0.5, mu=5.3, sg=0.01)
    mi.limits["f"] = (0.0, 1.0)
    mi.limits["la"] = (0.01, 10)
    mi.limits["mu"] = (5.0, 5.6)
    mi.limits["sg"] = (0.001, 0.6)
    mi.migrad()
    param_ci_upper_h1 = np.array(mi.values) + np.array(mi.errors)
    param_ci_lower_h1 = np.array(mi.values) - np.array(mi.errors)
    param_true = np.array([0.1, 0.5, 5.28, 0.018])
    covered_h1 = np.sum(
        (param_ci_lower_h1 < param_true) & (param_true < param_ci_upper_h1)
    )
    valid_fit_h1 = mi.valid
    nll_sb_value = mi.fval

    # Perform binned ML fit using background only model.
    nll_b = BinnedNLL(nh, xe, cdf_h0_fast)
    mi = Minuit(nll_b, la=0.5)
    mi.limits["la"] = (0.01, 10)
    mi.migrad()
    covered_h0 = 1 * (
        (mi.values["la"] - mi.errors["la"] < 0.5)
        and (0.5 < mi.values["la"] + mi.errors["la"])
    )
    valid_fit_h0 = mi.valid
    nll_b_value = mi.fval

    valid_llr = valid_fit_h1 and valid_fit_h0

    # Return the log-likelihood ratio.
    return (nll_b_value - nll_sb_value), covered_h1, covered_h0, valid_llr


# Create global variables that enable fast inverse CDF sampling,
# sacrificing some memory overhead to enable fast processing.
GRANULAR = 5
input_space = np.linspace(5.0, 5.6, int(10**GRANULAR))

# Compute CDF values across interval [5.0, 5.6].
cdf_mixed_approx_h1 = cdf_norm_expon_mixed(
    input_space, 0.1, 0.5, 5.28, 0.018, 5.0, 5.6
)
quantiles_h1 = np.linspace(0.0, 1.0, int(10**GRANULAR))
cdf_mixed_inv_h1_01 = np.zeros(int(10**GRANULAR))
# For each quantile in [0.0, 1.0], find how far into the range [5.0, 5.6] you
# need to go in order for the cdf to equal the quantile.
for quantile_idx, quantile in enumerate(quantiles_h1):
    cdf_mixed_inv_h1_01[quantile_idx] = (10 ** (-GRANULAR)) * np.sum(
        cdf_mixed_approx_h1 < quantiles_h1[quantile_idx]
    )
# Transform into the interval [5.0, 5.6].
CDF_MIXED_INV_APPROX_H1 = 5.0 + 0.6 * cdf_mixed_inv_h1_01

# Repeat above for background only model.
cdf_mixed_approx_h0 = cdf_norm_expon_mixed(
    input_space, 0.0, 0.5, 5.28, 0.018, 5.0, 5.6
)
quantiles_h0 = np.linspace(0.0, 1.0, int(10**GRANULAR))
cdf_mixed_inv_h0_01 = np.zeros(int(10**GRANULAR))
for quantile_idx, quantile in enumerate(quantiles_h0):
    cdf_mixed_inv_h0_01[quantile_idx] = (10 ** (-GRANULAR)) * np.sum(
        cdf_mixed_approx_h0 < quantiles_h0[quantile_idx]
    )
CDF_MIXED_INV_APPROX_H0 = 5.0 + 0.6 * cdf_mixed_inv_h0_01


# Inverse cdf sampling of signal+background model.
def sample_h1_fast(sample_size, seed):
    g = np.random.default_rng(seed=seed)
    uniform_sample = g.integers(low=0, high=(10**GRANULAR), size=sample_size)
    sample = CDF_MIXED_INV_APPROX_H1[uniform_sample]
    return sample


# Inverse cdf sampling of background only model.
def sample_h0_fast(sample_size, seed):
    g = np.random.default_rng(seed=seed)
    uniform_sample = g.integers(low=0, high=(10**GRANULAR), size=sample_size)
    sample = CDF_MIXED_INV_APPROX_H0[uniform_sample]
    return sample


# Implement faster cdf functions so that minuit fits faster in simulation
# study.
def cdf_h1_fast(x, f, la, mu, sg):
    return (f * truncnorm.cdf(x, 5.0, 5.6, loc=mu, scale=sg)) + (
        (1 - f) * truncexpon.cdf(x, 5.0, 5.6, loc=0.0, scale=1 / la)
    )


def cdf_h0_fast(x, la):
    return truncexpon.cdf(x, 5.0, 5.6, loc=0.0, scale=1 / la)


def chi2_pdf(x, dof):
    return chi2.pdf(x, dof)


def T_simulation(N, N_toys):
    T_sb_sample = np.zeros(N_toys)
    T_b_sample = np.zeros(N_toys)
    covered_h1 = np.zeros(N_toys)
    covered_h0 = np.zeros(N_toys)
    valid_toys = np.zeros(N_toys)
    for toy_idx in range(N_toys):
        sample_array_sb = sample_h1_fast(N, toy_idx)
        T_sb_sample[toy_idx], covered_h1_toy, _, valid_llr_sb = compute_llr(
            sample_array_sb, h0_bool=False
        )

        sample_array_b = sample_h0_fast(N, toy_idx)
        T_b_sample[toy_idx], _, covered_h0_toy, valid_llr_b = compute_llr(
            sample_array_b, h0_bool=True
        )

        covered_h1[toy_idx] = covered_h1_toy
        covered_h0[toy_idx] = covered_h0_toy
        valid_toys[toy_idx] = 1 * (valid_llr_sb and valid_llr_b)

    # Filter to only use data for when all the minuit fits in the thrown toy
    # converged to a valid minimum.
    valid_toys_total = np.sum(valid_toys)
    T_sb_sample = T_sb_sample[valid_toys == 1]
    T_b_sample = T_b_sample[valid_toys == 1]
    covered_total_h1 = np.sum(covered_h1[valid_toys == 1])
    covered_total_h0 = np.sum(covered_h0[valid_toys == 1])

    nll = UnbinnedNLL(T_b_sample, chi2_pdf)
    mi = Minuit(nll, dof=3.0)
    mi.limits["dof"] = (0.0, 10.0)
    mi.migrad()
    dof_T_b = mi.values["dof"]
    T0 = chi2.ppf(1 - (2.9e-7), dof_T_b)
    power = np.sum(T_sb_sample > T0) / T_sb_sample.shape[0]
    return T0, power, covered_total_h1, covered_total_h0, valid_toys_total
