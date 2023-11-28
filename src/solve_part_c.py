import numpy as np
from scipy.stats import norm, expon
from scipy.special import erf
from scipy.integrate import quad


def pdf_norm_expon_mixed(x, f, la, mu, sg, alpha, beta):
    pdf_s = norm.pdf(x, loc=mu, scale=sg)
    pdf_b = expon.pdf(x, loc=0, scale=1/la)
    weight_s = (2 * f) / ( erf( (beta - mu) / (sg*np.sqrt(2)) ) - erf( (alpha - mu) / (sg*np.sqrt(2)) ) )
    weight_b = (1 - f) / (np.exp( - la * alpha) - np.exp( - la * beta))
    return (weight_s * pdf_s) + (weight_b * pdf_b)


def check_normalisation(pdf, lower, upper):
    integral_value, abserr = quad(pdf, lower, upper)
    return integral_value

pdf_1 = lambda x: pdf_norm_expon_mixed(x, f=0.05, la=0.5, mu=5.4, sg=0.1, alpha=5.0, beta=5.6)
pdf_2 = lambda x: pdf_norm_expon_mixed(x, f=0.9, la=0.5, mu=5.4, sg=0.01, alpha=5.0, beta=5.6)
pdf_3 = lambda x: pdf_norm_expon_mixed(x, f=0.1, la=2.0, mu=1.0, sg=1.0, alpha=5.0, beta=5.6)

print("Integral when 1st set of parameters used:    {}".format(check_normalisation(pdf_1, 5.0, 5.6)))
print("Integral when 2nd set of parameters used:    {}".format(check_normalisation(pdf_2, 5.0, 5.6)))
print("Integral when 3rd set of parameters used:    {}".format(check_normalisation(pdf_3, 5.0, 5.6)))