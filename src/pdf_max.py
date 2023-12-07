from mixed_pdf_tools import pdf_norm_expon_mixed
import numpy as np

sample_points = np.linspace(5.0, 5.6, 1000001)
print(
    "Maximum of signal+background hypothesis pdf =",
    np.max(
        pdf_norm_expon_mixed(sample_points, 0.1, 0.5, 5.28, 0.018, 5.0, 5.6)
    ),
)
print(
    "Maximum of background only hypothesis pdf =",
    np.max(
        pdf_norm_expon_mixed(sample_points, 0.0, 0.5, 5.28, 0.018, 5.0, 5.6)
    ),
)
