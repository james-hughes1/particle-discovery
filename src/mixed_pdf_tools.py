import numpy as np
from scipy.stats import norm, expon
from scipy.special import erf
from scipy.integrate import quad
import matplotlib.pyplot as plt


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
    f, la, mu, sg, alpha, beta, filename, npoints=1001, sample_hist=None
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
    ax.plot(x_plot, f * scale_factor * pdf_s)
    ax.plot(x_plot, (1 - f) * scale_factor * pdf_b)
    ax.plot(x_plot, scale_factor * pdf_sb)
    plt.savefig("outputs/" + filename)


def rej_sample(pdf, sample_size, lower, upper, max_jobs):
    x_vals = np.linspace(lower, upper, 100001)
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
