import numpy as np
from numba_stats import norm, expon
from scipy.special import erf
from scipy.integrate import quad


def pdf_norm_expon_mixed(x, f, la, mu, sg, alpha, beta):
    # Compute the value signal and background pdf values without
    # truncating to [alpha, beta].
    pdf_s = norm.pdf(x, loc=mu, scale=sg)
    pdf_b = expon.pdf(x, loc=0, scale=1 / la)
    # Compute the pdf value weights, including the signal fraction f
    # and normalising to the interval.
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2)))
        - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))
    return (weight_s * pdf_s) + (weight_b * pdf_b)


def cdf_norm_expon_mixed(x, f, la, mu, sg, alpha, beta):
    # Compute the value signal and background cdf values wihtout
    # truncating to [alpha, beta].
    cdf_s = norm.cdf(x, loc=mu, scale=sg) - norm.cdf(
        alpha * np.ones(x.shape), loc=mu, scale=sg
    )
    cdf_b = expon.cdf(x, loc=0, scale=1 / la) - expon.cdf(
        alpha * np.ones(x.shape), loc=0, scale=1 / la
    )
    # Compute the cdf value weights.
    weight_s = (2 * f) / (
        erf((beta - mu) / (sg * np.sqrt(2)))
        - erf((alpha - mu) / (sg * np.sqrt(2)))
    )
    weight_b = (1 - f) / (np.exp(-la * alpha) - np.exp(-la * beta))
    return (weight_s * cdf_s) + (weight_b * cdf_b)


def check_normalisation(pdf, lower, upper):
    integral_value, abserr = quad(pdf, lower, upper)
    return integral_value
