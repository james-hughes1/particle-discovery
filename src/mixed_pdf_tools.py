import numpy as np
from scipy.stats import norm, expon
from scipy.special import erf
from scipy.integrate import quad
import matplotlib.pyplot as plt


def pdf_norm_expon_mixed(x, f, la, mu, sg, alpha, beta):
    pdf_s = norm.pdf(x, loc=mu, scale=sg)
    pdf_b = expon.pdf(x, loc=0, scale=1/la)
    weight_s = (2 * f) / (erf((beta - mu) / (sg*np.sqrt(2))) - erf((alpha - mu) / (sg*np.sqrt(2))))
    weight_b = (1 - f) / (np.exp(- la * alpha) - np.exp(- la * beta))
    return (weight_s * pdf_s) + (weight_b * pdf_b)


def check_normalisation(pdf, lower, upper):
    integral_value, abserr = quad(pdf, lower, upper)
    return integral_value

def plot_signal_background_mixed(f, la, mu, sg, alpha, beta, filename, npoints=1001):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    x_plot = np.linspace(alpha, beta, npoints)
    pdf_s = norm.pdf(x_plot, loc=mu, scale=sg)
    pdf_b = expon.pdf(x_plot, loc=0, scale=1/la)
    pdf_sb = pdf_norm_expon_mixed(x_plot, f, la, mu, sg, alpha, beta)
    ax.plot(x_plot, pdf_s)
    ax.plot(x_plot, pdf_b)
    ax.plot(x_plot, pdf_sb)
    plt.savefig("plots/"+filename)


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
