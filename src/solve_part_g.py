from simulation_tools import find_sample_size

# Run simulation studies with increasing granularity on the sample sizes used.
find_sample_size(N_toys=1000, N_min=2000, N_max=3050, N_step=50, model="g")
find_sample_size(N_toys=20000, N_min=2550, N_max=2670, N_step=20, model="g")
