from mixed_pdf_tools import rej_sample_h0_fast, rej_sample_h1_fast, compute_llr
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from line_profiler import LineProfiler


def T_simulation(N, N_toys):
    T_sb_sample = np.zeros(N_toys)
    T_b_sample = np.zeros(N_toys)
    for toy_idx in tqdm(range(N_toys)):
        sample_array_sb = rej_sample_h1_fast(N, 10 * N)
        T_sb_sample[toy_idx] = compute_llr(sample_array_sb)

        sample_array_b = rej_sample_h0_fast(N, 10 * N)
        T_b_sample[toy_idx] = compute_llr(sample_array_b)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(np.concatenate([T_sb_sample, T_b_sample]))
    plt.savefig("outputs/T_simulation.png")


lp = LineProfiler()
lp_wrapper = lp(T_simulation)
lp_wrapper(10000, 10)
lp.print_stats()
