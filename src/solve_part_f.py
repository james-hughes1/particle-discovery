from mixed_pdf_tools import rej_sample, pdf_h0_fast, pdf_h1_fast, compute_llr
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

N = 10000
N_toys = 1000
T_sb_sample = np.zeros(N_toys)
T_b_sample = np.zeros(N_toys)
for toy_idx in tqdm(range(N_toys)):
    sample_array_sb, _ = rej_sample(pdf_h1_fast, N, 5.0, 5.6, 10 * N, 3.73)
    T_sb_sample[toy_idx] = compute_llr(sample_array_sb)

    sample_array_b, _ = rej_sample(pdf_h0_fast, N, 5.0, 5.6, 10 * N, 1.93)
    T_b_sample[toy_idx] = compute_llr(sample_array_b)


fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(np.concatenate([T_sb_sample, T_b_sample]))
plt.savefig("outputs/T_simulation.png")
