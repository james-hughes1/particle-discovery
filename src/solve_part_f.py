from mixed_pdf_tools import T_simulation
import time
from tqdm import tqdm

# Set simulation parameters
N_toys = 5000
N_min, N_max, N_step = 700, 800, 10

filename = (
    "simulation_study_"
    + str(N_min)
    + "_"
    + str(N_max)
    + "_"
    + str(N_step)
    + "_"
    + str(N_toys)
    + ".txt"
)


# Run simulation
N_list = []
T0_list = []
power_list = []
start = time.time()
for N in tqdm(range(N_min, N_max, N_step)):
    N_list.append(N)
    T0, power = T_simulation(N, N_toys)
    T0_list.append(T0)
    power_list.append(power)
end = time.time()
exec_time = end - start


with open("data/" + filename, "w") as f:
    f.write(
        "N_min: {}  N_max: {}   N_step: {}  N_toys: {}\n".format(
            N_min, N_max, N_step, N_toys
        )
    )
    f.write("Execution time = {}s\n".format(exec_time))
    for idx in range(len(N_list)):
        f.write(
            "N: {:>5}   |   T0: {:.6f}, Power: {:.6f}\n".format(
                N_list[idx], T0_list[idx], power_list[idx]
            )
        )
