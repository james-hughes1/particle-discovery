import numpy as np
from iminuit import Minuit
from iminuit.cost import BinnedNLL

from mixed_pdf_tools import (
    pdf_norm_expon_mixed,
    rej_sample,
    cdf_norm_expon_mixed,
    plot_signal_background_mixed,
)


# Produce the sample, and separate into bins
def pdf(x):
    pdf_norm_expon_mixed(x, 0.1, 0.5, 5.28, 0.018, 5.0, 5.6)


sample_array = rej_sample(pdf, 100000, 5.0, 5.6, 100000000)[0]
nh, xe = np.histogram(sample_array, bins=100, range=(5, 5.6))
cx = 0.5 * (xe[1:] + xe[:-1])

# Perform binned ML fit of parameters.
nll = BinnedNLL(nh, xe, cdf_norm_expon_mixed)
mi = Minuit(nll, f=0.5, la=0.5, mu=5.3, sg=0.1, alpha=5.0, beta=5.6)
mi.fixed["alpha"] = True
mi.fixed["beta"] = True
mi.limits["f"] = (0, 1)
mi.limits["mu"] = (5.0, 5.6)
mi.limits["sg"] = (1e-9, 10)

print(mi.migrad())
plot_signal_background_mixed(
    *mi.values, "part_e_plot.png", sample_hist=(nh, xe)
)
