import numpy as np
from numba_stats import norm, expon, truncnorm, truncexpon
from scipy.special import erf
from scipy.integrate import quad
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import BinnedNLL, UnbinnedNLL
from scipy.stats import chi2
import time
from tqdm import tqdm


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


def compute_llr_f(sample_array):
    # Separate the sample into bins.
    N = len(sample_array)
    nh, xe = np.histogram(sample_array, bins=int(N / 10), range=(5, 5.6))

    # Perform binned ML fit of parameters using signal+background model.
    nll_sb = BinnedNLL(nh, xe, cdf_sb_fast)
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
    nll_b = BinnedNLL(nh, xe, cdf_b_fast)
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
cdf_mixed_approx_sb = cdf_norm_expon_mixed(
    input_space, 0.1, 0.5, 5.28, 0.018, 5.0, 5.6
)
quantiles_sb = np.linspace(0.0, 1.0, int(10**GRANULAR))
cdf_mixed_inv_sb_01 = np.zeros(int(10**GRANULAR))
# For each quantile in [0.0, 1.0], find how far into the range [5.0, 5.6] you
# need to go in order for the cdf to equal the quantile.
for quantile_idx, quantile in enumerate(quantiles_sb):
    cdf_mixed_inv_sb_01[quantile_idx] = (10 ** (-GRANULAR)) * np.sum(
        cdf_mixed_approx_sb < quantiles_sb[quantile_idx]
    )
# Transform into the interval [5.0, 5.6].
CDF_MIXED_INV_APPROX_SB = 5.0 + 0.6 * cdf_mixed_inv_sb_01

# Repeat above for background only model.
cdf_mixed_approx_b = cdf_norm_expon_mixed(
    input_space, 0.0, 0.5, 5.28, 0.018, 5.0, 5.6
)
quantiles_b = np.linspace(0.0, 1.0, int(10**GRANULAR))
cdf_mixed_inv_b_01 = np.zeros(int(10**GRANULAR))
for quantile_idx, quantile in enumerate(quantiles_b):
    cdf_mixed_inv_b_01[quantile_idx] = (10 ** (-GRANULAR)) * np.sum(
        cdf_mixed_approx_b < quantiles_b[quantile_idx]
    )
CDF_MIXED_INV_APPROX_B = 5.0 + 0.6 * cdf_mixed_inv_b_01


# Inverse cdf sampling of signal+background model.
def sample_sb_fast(sample_size, seed):
    g = np.random.default_rng(seed=seed)
    uniform_sample = g.integers(low=0, high=(10**GRANULAR), size=sample_size)
    sample = CDF_MIXED_INV_APPROX_SB[uniform_sample]
    return sample


# Inverse cdf sampling of background only model.
def sample_b_fast(sample_size, seed):
    g = np.random.default_rng(seed=seed)
    uniform_sample = g.integers(low=0, high=(10**GRANULAR), size=sample_size)
    sample = CDF_MIXED_INV_APPROX_B[uniform_sample]
    return sample


# Implement faster cdf functions so that minuit fits faster in simulation
# study.
def cdf_sb_fast(x, f, la, mu, sg):
    return (f * truncnorm.cdf(x, 5.0, 5.6, loc=mu, scale=sg)) + (
        (1 - f) * truncexpon.cdf(x, 5.0, 5.6, loc=0.0, scale=1 / la)
    )


def cdf_b_fast(x, la):
    return truncexpon.cdf(x, 5.0, 5.6, loc=0.0, scale=1 / la)


def chi2_pdf(x, dof):
    return chi2.pdf(x, dof)


def cdf_2sb_fast(x, f1, f2, la, mu1, mu2, sg):
    return (
        (f1 * truncnorm.cdf(x, 5.0, 5.6, loc=mu1, scale=sg))
        + (f2 * truncnorm.cdf(x, 5.0, 5.6, loc=mu2, scale=sg))
        + ((1 - f1 - f2) * truncexpon.cdf(x, 5.0, 5.6, loc=0.0, scale=1 / la))
    )


# Repeat fast cdf inverse sampling for 2 signal + background model.
cdf_mixed_approx_2sb = cdf_2sb_fast(
    input_space, 0.1, 0.05, 0.5, 5.28, 5.35, 0.018
)
quantiles_2sb = np.linspace(0.0, 1.0, int(10**GRANULAR))
cdf_mixed_inv_2sb_01 = np.zeros(int(10**GRANULAR))
for quantile_idx, quantile in enumerate(quantiles_2sb):
    cdf_mixed_inv_2sb_01[quantile_idx] = (10 ** (-GRANULAR)) * np.sum(
        cdf_mixed_approx_2sb < quantiles_2sb[quantile_idx]
    )
CDF_MIXED_INV_APPROX_2SB = 5.0 + 0.6 * cdf_mixed_inv_2sb_01


def sample_2sb_fast(sample_size, seed):
    g = np.random.default_rng(seed=seed)
    uniform_sample = g.integers(low=0, high=(10**GRANULAR), size=sample_size)
    sample = CDF_MIXED_INV_APPROX_2SB[uniform_sample]
    return sample


def compute_llr_g(sample_array):
    # Separate the sample into bins.
    N = len(sample_array)
    nh, xe = np.histogram(sample_array, bins=int(N / 10), range=(5, 5.6))

    # Perform binned ML fit using 2 signal + background model.
    nll_2sb = BinnedNLL(nh, xe, cdf_2sb_fast)
    mi = Minuit(
        nll_2sb, f1=0.05, f2=0.05, la=0.5, mu1=5.28, mu2=5.35, sg=0.018
    )
    mi.limits["f1"] = (0.0, 1.0)
    mi.limits["f2"] = (0.0, 1.0)
    mi.limits["la"] = (0.01, 10)
    mi.limits["mu1"] = (5.0, 5.6)
    mi.limits["mu2"] = (5.0, 5.6)
    mi.limits["sg"] = (0.001, 0.6)
    mi.migrad()
    covered_h0 = 1 * (
        (mi.values["la"] - mi.errors["la"] < 0.5)
        and (0.5 < mi.values["la"] + mi.errors["la"])
    )
    param_ci_upper_h1 = np.array(mi.values) + np.array(mi.errors)
    param_ci_lower_h1 = np.array(mi.values) - np.array(mi.errors)
    param_true_h1 = np.array([0.1, 0.05, 0.5, 5.28, 5.35, 0.018])
    covered_h1 = np.sum(
        (param_ci_lower_h1 < param_true_h1)
        & (param_true_h1 < param_ci_upper_h1)
    )
    valid_fit_h1 = mi.valid
    nll_2sb_value = mi.fval

    # Perform binned ML fit of parameters using signal+background model.
    nll_sb = BinnedNLL(nh, xe, cdf_sb_fast)
    mi = Minuit(nll_sb, f=0.1, la=0.5, mu=5.3, sg=0.018)
    mi.limits["f"] = (0.0, 1.0)
    mi.limits["la"] = (0.01, 10)
    mi.limits["mu"] = (5.0, 5.6)
    mi.limits["sg"] = (0.001, 0.6)
    mi.migrad()
    param_ci_upper_h0 = np.array(mi.values) + np.array(mi.errors)
    param_ci_lower_h0 = np.array(mi.values) - np.array(mi.errors)
    param_true_h0 = np.array([0.1, 0.5, 5.28, 0.018])
    covered_h0 = np.sum(
        (param_ci_lower_h0 < param_true_h0)
        & (param_true_h0 < param_ci_upper_h0)
    )
    valid_fit_h0 = mi.valid
    nll_sb_value = mi.fval

    valid_llr = valid_fit_h0 and valid_fit_h1

    # Return the log-likelihood ratio.
    return (nll_sb_value - nll_2sb_value), covered_h1, covered_h0, valid_llr


def T_simulation(N, N_toys, model, plot):
    assert model in ["f", "g"]
    T_h1_sample = np.zeros(N_toys)
    T_h0_sample = np.zeros(N_toys)
    covered_h1 = np.zeros(N_toys)
    covered_h0 = np.zeros(N_toys)
    valid_toys = np.zeros(N_toys)
    for toy_idx in range(N_toys):
        if model == "f":
            # Generate a llr value for a signal background sample and a
            # background only sample.
            sample_array_sb = sample_sb_fast(N, toy_idx)
            (
                T_h1_sample[toy_idx],
                covered_h1_toy,
                _,
                valid_llr_h1,
            ) = compute_llr_f(sample_array_sb)

            sample_array_b = sample_b_fast(N, toy_idx)
            (
                T_h0_sample[toy_idx],
                _,
                covered_h0_toy,
                valid_llr_h0,
            ) = compute_llr_f(sample_array_b)
        else:
            # Generate a llr value for a 2 signal + background sample and a
            # 1 signal + background sample.
            sample_array_2sb = sample_2sb_fast(N, toy_idx)
            (
                T_h1_sample[toy_idx],
                covered_h1_toy,
                _,
                valid_llr_h1,
            ) = compute_llr_g(sample_array_2sb)

            sample_array_sb = sample_sb_fast(N, toy_idx)
            (
                T_h0_sample[toy_idx],
                _,
                covered_h0_toy,
                valid_llr_h0,
            ) = compute_llr_g(sample_array_sb)
        covered_h1[toy_idx] = covered_h1_toy
        covered_h0[toy_idx] = covered_h0_toy
        valid_toys[toy_idx] = 1 * (valid_llr_h1 and valid_llr_h0)

    # Filter to only use data for when all the minuit fits in the thrown toy
    # converged to a valid minimum.
    valid_toys_total = np.sum(valid_toys)
    T_h1_sample = T_h1_sample[valid_toys == 1]
    T_h0_sample = T_h0_sample[valid_toys == 1]
    covered_total_h1 = np.sum(covered_h1[valid_toys == 1])
    covered_total_h0 = np.sum(covered_h0[valid_toys == 1])

    nll = UnbinnedNLL(T_h0_sample, chi2_pdf)
    mi = Minuit(nll, dof=3.0)
    mi.limits["dof"] = (0.0, 10.0)
    mi.migrad()
    dof_T_b = mi.values["dof"]
    T0 = chi2.ppf(1 - (2.9e-7), dof_T_b)
    power = np.sum(T_h1_sample > T0) / T_h1_sample.shape[0]

    # Plot the simulated distributions of T.
    if plot:
        x_plot = np.linspace(0, np.max(T_h1_sample), 201)
        fig, ax = plt.subplots()
        ax.plot(
            x_plot,
            N_toys * chi2_pdf(x_plot, dof_T_b),
            label="Chi2 density estimate for H0",
            c="blue",
        )
        ax.hist(
            T_h1_sample,
            bins=10,
            label="Distribution under H1",
            alpha=0.6,
            color="red",
        )
        ax.hist(
            T_h0_sample,
            bins=10,
            label="Distribution under H0",
            alpha=0.6,
            color="blue",
        )
        plt.axvline(
            T0,
            label="Critical T0 for discovery",
            color="black",
            linestyle="--",
        )
        ax.legend()
        ax.set(
            title=f"Simulated distributions of T: model={model}, N={N}, "
            f"N_toys={N_toys}"
        )
        plt.savefig(f"outputs/T_distributions_{model}_{N}_{N_toys}.png")
    return T0, power, covered_total_h1, covered_total_h0, valid_toys_total


def find_sample_size(N_toys, N_min, N_max, N_step, model):
    # Save to a filename indicating which parameters were used.
    filename = (
        "simulation_study_"
        + str(model)
        + "_"
        + str(N_min)
        + "_"
        + str(N_max)
        + "_"
        + str(N_step)
        + "_"
        + str(N_toys)
        + ".txt"
    )

    # Run simulation.
    N_list = []
    T0_list = []
    power_list = []
    coverage_h1 = []
    coverage_h0 = []
    valid_sims = []
    start = time.time()
    for N in tqdm(range(N_min, N_max, N_step)):
        N_list.append(N)
        # Plot on the last simulation.
        (
            T0,
            power,
            covered_total_h1,
            covered_total_h0,
            valid_toys,
        ) = T_simulation(N, N_toys, model, (N == N_max - N_step))
        T0_list.append(T0)
        power_list.append(power)
        # Adjust the average coverage figures for the number of parameters
        # being estimated.
        if model == "f":
            coverage_h1.append(covered_total_h1 / (4 * valid_toys))
            coverage_h0.append(covered_total_h0 / valid_toys)
        else:
            coverage_h1.append(covered_total_h1 / (6 * valid_toys))
            coverage_h0.append(covered_total_h0 / (4 * valid_toys))
        valid_sims.append(valid_toys / N_toys)
    end = time.time()
    exec_time = end - start

    # Write simulation study data to file.
    with open("outputs/" + filename, "w") as f:
        f.write("Model: {}\n".format(model))
        f.write(
            "N_min: {}  N_max: {}   N_step: {}  N_toys: {}\n".format(
                N_min, N_max, N_step, N_toys
            )
        )
        f.write("Execution time = {:.3f}s\n\n\n".format(exec_time))
        f.write(
            "{:>5}||{:<8}|{:<8}|{:<8}|{:<9}|{:<11}\n".format(
                "N", "C. H1", "C. H0", "Valid", "T0", "Power"
            )
        )
        f.write("-----++--------+--------+--------+---------+-----------\n")
        sim_data_format_str = (
            "{:>5}||{:<8.3f}|{:<8.3f}|{:<8.3f}|{:<9.3f}|{:<11.6f}\n"
        )
        for sim_data in zip(
            N_list, coverage_h1, coverage_h0, valid_sims, T0_list, power_list
        ):
            f.write(sim_data_format_str.format(*sim_data))
