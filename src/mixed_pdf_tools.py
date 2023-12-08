import numpy as np
from numba_stats import norm, expon
from scipy.special import erf
from scipy.integrate import quad
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import BinnedNLL


def pdf_norm_expon_mixed(x, f, la, mu, sg, alpha, beta):
    pdf_s = norm.pdf(x, loc=mu, scale=sg)
    pdf_b = expon.pdf(x, loc=0, scale=1 / la)
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2)))
        - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))
    print(pdf_b)
    print(
        "Here:",
        expon.pdf(x, loc=0, scale=1 / la),
        la,
        expon.pdf(x, loc=0.0, scale=5.0),
    )
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
    filename,
    alpha=None,
    beta=None,
    npoints=1001,
    sample_hist=None,
):
    if alpha is None:
        alpha, beta = 5.0, 5.6
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
    pdf_sb = pdf_h1_fast(x_plot, f, la, mu, sg)
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2)))
        - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))
    ax.plot(x_plot, weight_s * scale_factor * pdf_s)
    ax.plot(x_plot, weight_b * scale_factor * pdf_b)
    ax.plot(x_plot, scale_factor * pdf_sb)
    plt.savefig("outputs/" + filename)


def rej_sample(pdf, sample_size, lower, upper, max_jobs, y_max=None):
    x_vals = np.linspace(lower, upper, 100001)
    if y_max is None:
        y_max = np.max(pdf(x_vals))
    sample = []
    jobs = 0
    while len(sample) < sample_size and jobs < max_jobs:
        x = np.random.uniform(lower, upper)
        y = np.random.uniform(0, y_max)
        if y < pdf(x):
            sample.append(x)
        jobs += 1
    return sample, jobs


WEIGHT_B_H0 = 1.0 / (np.exp(-0.5 * 5.0) - np.exp(-0.5 * 5.6))


def pdf_h0_fast(x, la=None):
    if la is None:
        return WEIGHT_B_H0 * expon.pdf(x, loc=0, scale=2.0)
    else:
        weight_b = 1.0 / (np.exp(-la * 5.0) - np.exp(-la * 5.6))
        pdf_b = expon.pdf(x, loc=0, scale=1 / la)
        return weight_b * pdf_b


def cdf_h0_fast(x, la=None):
    if la is None:
        cdf_b = expon.cdf(x, loc=0, scale=2.0) - expon.cdf(
            5.0, loc=0, scale=2.0
        )
        return WEIGHT_B_H0 * cdf_b
    else:
        weight_b = 1.0 / (np.exp(-la * 5.0) - np.exp(-la * 5.6))
        cdf_b = expon.cdf(x, loc=0, scale=1 / la) - expon.cdf(
            5.0, loc=0, scale=1 / la
        )
        return weight_b * cdf_b


WEIGHT_S_H1 = 0.2 / (
    erf(0.32 / (0.018 * np.sqrt(2))) - erf(-0.28 / (0.18 * np.sqrt(2)))
)
WEIGHT_B_H1 = 0.9 / (np.exp(-0.5 * 5.0) - np.exp(-0.5 * 5.6))


def pdf_h1_fast(x, f=None, la=None, mu=None, sg=None):
    if la is None:
        return (WEIGHT_S_H1 * norm.pdf(x, loc=5.28, scale=0.018)) + (
            WEIGHT_B_H1 * expon.pdf(x, loc=0, scale=2.0)
        )
    else:
        weight_s = (2 * f) / (
            erf((5.6 - mu) / (sg * np.sqrt(2)))
            - erf((5.0 - mu) / (sg * np.sqrt(2)))
        )
        weight_b = (1 - f) / (np.exp(-la * 5.0) - np.exp(-la * 5.6))
        pdf_s = norm.pdf(x, loc=mu, scale=sg)
        pdf_b = expon.pdf(x, loc=0, scale=1 / la)
        return (weight_s * pdf_s) + (weight_b * pdf_b)


def cdf_h1_fast(x, f=None, la=None, mu=None, sg=None):
    if la is None:
        cdf_s = norm.cdf(x, loc=5.28, scale=0.018) - norm.cdf(
            5.0, loc=5.28, scale=0.018
        )
        cdf_b = expon.cdf(x, loc=0, scale=2.0) - expon.cdf(
            5.0, loc=0, scale=2.0
        )
        return (WEIGHT_S_H1 * cdf_s) + (WEIGHT_B_H1 * cdf_b)
    else:
        weight_s = (2 * f) / (
            erf((5.6 - mu) / (sg * np.sqrt(2)))
            - erf((5.0 - mu) / (sg * np.sqrt(2)))
        )
        weight_b = (1 - f) / (np.exp(-la * 5.0) - np.exp(-la * 5.6))
        cdf_s = norm.cdf(x, loc=mu, scale=sg) - norm.cdf(5.0, loc=mu, scale=sg)
        cdf_b = expon.cdf(x, loc=0, scale=1 / la) - expon.cdf(
            5.0, loc=0, scale=1 / la
        )
        return (weight_s * cdf_s) + (weight_b * cdf_b)


def compute_llr(sample_array):
    # Separate the sample into bins.
    N = len(sample_array)
    nh, xe = np.histogram(sample_array, bins=int(N / 1000), range=(5, 5.6))

    # Perform binned ML fit of parameters using signal+background model.
    nll_sb = BinnedNLL(nh, xe, cdf_h1_fast)
    mi = Minuit(nll_sb, f=0.1, la=0.5, mu=5.3, sg=0.1)
    mi.limits["f"] = (0, 1)
    mi.limits["la"] = (1e-9, 1e2)
    mi.limits["mu"] = (5.0, 5.6)
    mi.limits["sg"] = (1e-9, 10)
    mi.migrad()
    nll_sb_value = mi.fval

    # Perform binned ML fit using background only model.
    nll_b = BinnedNLL(nh, xe, cdf_h0_fast)
    mi = Minuit(nll_b, la=0.5)
    mi.limits["la"] = (1e-9, 1e2)
    mi.migrad()
    nll_b_value = mi.fval

    # Return the log-likelihood ratio.
    return nll_b_value - nll_sb_value


GRANULAR = 5
input_space = np.linspace(5.0, 5.6, int(10**GRANULAR))

cdf_mixed_approx_h1 = cdf_norm_expon_mixed(
    input_space, 0.1, 0.5, 5.28, 0.018, 5.0, 5.6
)
quantiles_h1 = np.linspace(0.0, 1.0, int(10**GRANULAR))
cdf_mixed_inv_h1_01 = np.zeros(int(10**GRANULAR))
for quantile_idx, quantile in enumerate(quantiles_h1):
    cdf_mixed_inv_h1_01[quantile_idx] = (10 ** (-GRANULAR)) * np.sum(
        cdf_mixed_approx_h1 < quantiles_h1[quantile_idx]
    )
CDF_MIXED_INV_APPROX_H1 = 5.0 + 0.6 * cdf_mixed_inv_h1_01

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


def sample_h1_fast(sample_size):
    g = np.random.default_rng(seed=1)
    uniform_sample = g.integers(low=0, high=(10**GRANULAR), size=sample_size)
    sample = CDF_MIXED_INV_APPROX_H1[uniform_sample]
    return sample


def sample_h0_fast(sample_size):
    g = np.random.default_rng(seed=2)
    uniform_sample = g.integers(low=0, high=(10**GRANULAR), size=sample_size)
    sample = CDF_MIXED_INV_APPROX_H0[uniform_sample]
    return sample
