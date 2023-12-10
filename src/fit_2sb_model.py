import numpy as np
from iminuit import Minuit
from iminuit.cost import BinnedNLL
import matplotlib.pyplot as plt
from numba_stats import truncexpon, truncnorm

from mixed_pdf_tools import (
    sample_2sb_fast,
    cdf_2sb_fast,
)

sample_array = sample_2sb_fast(100000, seed=1)
nh, xe = np.histogram(sample_array, bins=100, range=(5, 5.6))

# Perform binned ML fit of parameters.
nll = BinnedNLL(nh, xe, cdf_2sb_fast)
mi = Minuit(nll, f1=0.1, f2=0.05, la=0.5, mu1=5.2, mu2=5.4, sg=0.02)
mi.limits["f1"] = (0, 1.0)
mi.limits["f2"] = (0, 1.0)
mi.limits["la"] = (1e-9, 1e2)
mi.limits["mu1"] = (5.0, 5.6)
mi.limits["mu2"] = (5.0, 5.6)
mi.limits["sg"] = (1e-9, 1e2)

print(mi.migrad())

f1, f2, la, mu1, mu2, sg = mi.values

# Plot sample bins and fit densities for signals and background.
fig, ax = plt.subplots(figsize=(10, 8))
x_plot = np.linspace(5.0, 5.6, 1001)
cx = 0.5 * (xe[1:] + xe[:-1])
ax.errorbar(cx, nh, nh**0.5, fmt="ko")
scale_factor = np.sum(nh) * (xe[1] - xe[0])
pdf_s1 = truncnorm.pdf(x_plot[:-1], 5.0, 5.6, loc=mu1, scale=sg)
pdf_s2 = truncnorm.pdf(x_plot[:-1], 5.0, 5.6, loc=mu2, scale=sg)
pdf_b = truncexpon.pdf(x_plot[:-1], 5.0, 5.6, loc=0, scale=1 / la)
pdf_2sb = f1 * pdf_s1 + f2 * pdf_s2 + (1 - f1 - f2) * pdf_b
ax.plot(x_plot[:-1], f1 * scale_factor * pdf_s1, label="Signal 1 component")
ax.plot(x_plot[:-1], f2 * scale_factor * pdf_s2, label="Signal 2 component")
ax.plot(
    x_plot[:-1],
    (1 - f1 - f2) * scale_factor * pdf_b,
    label="Background component",
)
ax.plot(x_plot[:-1], scale_factor * pdf_2sb, label="Fit pdf (2s+b model)")
ax.legend()
ax.set(
    title="Binned ML Fit for Sample of Size 100K Using 2 Signal + "
    "Background Model",
    xlabel="M",
    ylabel="Events",
)
plt.savefig("outputs/2sb_fit_plot.png")
