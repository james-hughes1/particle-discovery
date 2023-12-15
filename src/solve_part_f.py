from simulation_tools import find_sample_size

# Run simulation studies with increasing granularity on the sample sizes used.
find_sample_size(N_toys=1000, N_min=50, N_max=1050, N_step=50, model="f")
find_sample_size(N_toys=10000, N_min=700, N_max=755, N_step=5, model="f")
