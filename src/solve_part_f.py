from mixed_pdf_tools import T_simulation
import time


N_toys = 1000
N_list = []
T0_list = []
power_list = []
N_min, N_max, N_step = 900, 1000, 20
start = time.time()
for N in range(N_min, N_max, N_step):
    N_list.append(N)
    T0, power = T_simulation(N, N_toys)
    T0_list.append(T0)
    power_list.append(power)
end = time.time()
exec_time = end - start

filename = "simulation_study_f_1.txt"

with open("data/" + filename, "w") as f:
    f.write(
        "N_min: {}  N_max: {}   N_step: {}  N_toys: {}\n".format(
            N_min, N_max, N_step, N_toys
        )
    )
    f.write("Execution time = {}\n".format(exec_time))
    for idx in range(len(N_list)):
        f.write(
            "N: {:>5}   |   T0: {:.6f}, Power: {:.6f}\n".format(
                N_list[idx], T0_list[idx], power_list[idx]
            )
        )
