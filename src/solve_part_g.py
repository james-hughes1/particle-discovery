from mixed_pdf_tools import T_simulation
import time
from tqdm import tqdm

# Set simulation parameters.
N_toys = 5000
N_min, N_max, N_step = 2500, 2600, 10
model = "g"

filename = (
    "simulation_study_"
    + str(model)
    + "_"
    + str(N_min)
    + "_"
    + str(N_max)
    + "_"
    + str(N_step)
    + "_"
    + str(N_toys)
    + ".txt"
)


# Run simulation.
N_list = []
T0_list = []
power_list = []
coverage_h1 = []
coverage_h0 = []
valid_sims = []
start = time.time()
for N in tqdm(range(N_min, N_max, N_step)):
    N_list.append(N)
    # Plot on the last simulation.
    T0, power, covered_total_h1, covered_total_h0, valid_toys = T_simulation(
        N, N_toys, model, (N == N_max - N_step)
    )
    T0_list.append(T0)
    power_list.append(power)
    coverage_h1.append(covered_total_h1 / (6 * valid_toys))
    coverage_h0.append(covered_total_h0 / (4 * valid_toys))
    valid_sims.append(valid_toys / N_toys)
end = time.time()
exec_time = end - start


# Write simulation study data to file.
with open("outputs/" + filename, "w") as f:
    f.write("Model: {}\n".format(model))
    f.write(
        "N_min: {}  N_max: {}   N_step: {}  N_toys: {}\n".format(
            N_min, N_max, N_step, N_toys
        )
    )
    f.write("Execution time = {:.3f}s\n\n\n".format(exec_time))
    f.write(
        "{:>5}||{:<8}|{:<8}|{:<8}|{:<9}|{:<11}\n".format(
            "N", "C. H1", "C. H0", "Valid", "T0", "Power"
        )
    )
    f.write("-----++--------+--------+--------+---------+-----------\n")
    for sim_data in zip(
        N_list, coverage_h1, coverage_h0, valid_sims, T0_list, power_list
    ):
        f.write(
            "{:>5}||{:<8.3f}|{:<8.3f}|{:<8.3f}|{:<9.3f}|{:<11.6f}\n".format(
                *sim_data
            )
        )
