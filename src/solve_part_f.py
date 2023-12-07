from mixed_pdf_tools import rej_sample, pdf_norm_expon_mixed, compute_llr
import matplotlib.pyplot as plt

N = 10000
T_sb_sample = []
T_b_sample = []
for idx in range(10):
    sample_array_sb, _ = rej_sample(
        lambda x: pdf_norm_expon_mixed(
            x, f=0.1, la=0.5, mu=5.28, sg=0.018, alpha=5.0, beta=5.6
        ),
        N,
        5.0,
        5.6,
        10 * N,
    )
    T_sb_sample.append(compute_llr(sample_array_sb))

    sample_array_b, _ = rej_sample(
        lambda x: pdf_norm_expon_mixed(
            x, f=0.0, la=0.5, mu=5.28, sg=0.018, alpha=5.0, beta=5.6
        ),
        N,
        5.0,
        5.6,
        10 * N,
    )
    T_sb_sample.append(compute_llr(sample_array_b))

fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(T_sb_sample)
ax.hist(T_b_sample)
plt.savefig("outputs/T_simulation.png")
print(T_sb_sample)
print(T_b_sample)
