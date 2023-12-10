import numpy as np
from iminuit import Minuit
from iminuit.cost import BinnedNLL

from mixed_pdf_tools import (
    sample_h1_fast,
    cdf_norm_expon_mixed,
    plot_signal_background_mixed,
)

sample_array = sample_h1_fast(100000, seed=1)
nh, xe = np.histogram(sample_array, bins=100, range=(5, 5.6))

# Perform binned ML fit of parameters.
nll = BinnedNLL(nh, xe, cdf_norm_expon_mixed)
mi = Minuit(nll, f=0.5, la=1.0, mu=5.3, sg=0.02, alpha=5.0, beta=5.6)
mi.fixed["alpha"] = True
mi.fixed["beta"] = True
mi.limits["f"] = (0, 1)
mi.limits["la"] = (1e-9, 1e2)
mi.limits["mu"] = (5.0, 5.6)
mi.limits["sg"] = (1e-9, 1e2)

print(mi.migrad())
plot_signal_background_mixed(
    *mi.values, "part_e_plot.png", sample_hist=(nh, xe)
)
