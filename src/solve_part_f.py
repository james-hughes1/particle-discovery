from mixed_pdf_tools import sample_h0_fast, sample_h1_fast, compute_llr
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from line_profiler import LineProfiler


def T_simulation(N, N_toys):
    T_sb_sample = np.zeros(N_toys)
    T_b_sample = np.zeros(N_toys)
    for toy_idx in tqdm(range(N_toys)):
        sample_array_sb = sample_h1_fast(N)
        T_sb_sample[toy_idx] = compute_llr(sample_array_sb)

        sample_array_b = sample_h0_fast(N)
        T_b_sample[toy_idx] = compute_llr(sample_array_b)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(T_sb_sample)
    plt.savefig("outputs/T_H1_simulation.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(T_b_sample)
    plt.savefig("outputs/T_H0_simulation.png")


lp = LineProfiler()
lp_wrapper = lp(T_simulation)
lp_wrapper(100000, 100)
lp.print_stats()
