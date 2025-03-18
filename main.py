#main file for running benchmarking tests
from mb_bench_test_2 import MB_benchmark
print("Starting the main")
known_Results = -0.443147 #per site using bethe ansatz for isotropic heisenberg chain
therm_size = 100
# nqbits, depths,  nshots, noise_params =[epsilon, n_Errors], rnds, ansatz, observe, thermal_value, thermal_size, hardware=None):
#currently only works for nshots = 0 due to qaptiva limitations
sample_results = MB_benchmark([2, 3], [2, 3], [0.0001, -1], 3, 'RYA', "Heisenberg", known_Results*therm_size, therm_size, 'supercond')