import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 16})  # Set global font size

def plot_analysis(bmark_dict):
    #figure 1 with efficiency vs metric - algorithmic and hardware
    #figure 2 with efficieny vs problem size
    #figure 3 with error vs problem size
    #figure 4 efficiency vs problem types

    #results format below
    #bmark['metrics'] = [sim_results, sim_variance, abs_errors, projected_results, error]
    #bmark['resources'] = [log_algo_res, log_hw_res]
    #bmark['efficiency'] = [log_algo_eff, log_hw_eff]

    #begin figure 1a
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for ansatz, bmark in bmark_dict.items():
        sim_results, sim_variance, abs_errors, projected_results, error = bmark['metrics']
        log_algo_res, log_hw_res, log_algo_res_list, log_hw_res_list = bmark['resources']

        metric = 1 - np.array(abs_errors)
        log_metric = np.log10(metric)
        log_algo_eff = log_metric - log_algo_res_list
        ax.plot(log_metric, log_algo_eff, 'o-', label=ansatz)

    ax.set_xlabel('Log Metric')
    ax.set_ylabel('Log ALGO Efficiency')
    ax.set_title('ALGO Efficiency vs metric')
    ax.legend()
    ax.grid(True)

    #plt.savefig(f'efficiency_vs_resources_{ansatz}.pdf', bbox_inches='tight')
    plt.show()
    
    #begin figure 1b
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for ansatz, bmark in bmark_dict.items():
        sim_results, sim_variance, abs_errors, projected_results, error = bmark['metrics']
        log_algo_res, log_hw_res, log_algo_res_list, log_hw_res_list = bmark['resources']

        metric = 1 - np.array(abs_errors)
        log_metric = np.log10(metric)
        log_hw_eff = log_metric - log_hw_res_list
        ax.plot(log_metric, log_hw_eff, 'o-', label=ansatz)

    ax.set_xlabel('Log Metric')
    ax.set_ylabel('Log HW Efficiency')
    ax.set_title('HW Efficiency vs metric')
    ax.legend()
    ax.grid(True)

    #plt.savefig(f'efficiency_vs_resources_{ansatz}.pdf', bbox_inches='tight')
    plt.show()
    
    #begin figure 2 : efficiency vs problem size
    qubits, depths, known_size = bmark['given_params']
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for ansatz, bmark in bmark_dict.items():
        sim_results, sim_variance, abs_errors, projected_results, error = bmark['metrics']
        log_algo_res, log_hw_res, log_algo_res_list, log_hw_res_list = bmark['resources']

        metric = 1 - np.array(abs_errors)
        log_metric = np.log10(metric)
        log_algo_eff = log_metric - log_algo_res_list
        #log_hw_eff = log_metric - log_hw_res_list
        ax.plot(int(qubits), log_algo_eff, 'o-', label=ansatz)

    ax.set_xlabel('Problem Size')
    ax.set_ylabel('Log ALGO Efficiency')
    ax.set_title('ALGO Efficiency vs Problem Size')
    ax.legend()
    ax.grid(True)

    # Add secondary x-axis for depths
    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('Depth')
    secax.set_xticks(qubits)
    secax.set_xticklabels(depths)

    #plt.savefig(f'efficiency_vs_resources_{ansatz}.pdf', bbox_inches='tight')
    plt.show()

    #begin figure 3
    # Generate linear fit
    qubits, depths, known_size = bmark['given_params']

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    for ansatz, bmark in bmark_dict.items():
        sim_results, sim_variance, abs_errors, projected_results, error = bmark['metrics']
        log_algo_res, log_hw_res, log_algo_res_list, log_hw_res_list = bmark['resources']

        slope, intercept = projected_results['regression_coefficients']
        x_range = np.linspace(min(qubits), known_size * 1.1, 100)
        y_fit = slope * x_range + intercept

        # Plot linear fit
        plt.plot(x_range, y_fit, 'r--', label='Linear fit for ' + ansatz)

        # Mark extrapolated point
        plt.errorbar([known_size], [projected_results['extrapolated_value']],
                    yerr=[projected_results['extrapolated_error']], fmt='ro', markersize=10,
                    capsize=5, label='Extrapolated value with error')

        # Add vertical line at target size
        plt.axvline(x=known_size, color='k', linestyle='--', alpha=0.5,
                    label=f'Target size: {known_size}')

    # Add labels, title, and legend
    plt.xlabel('Problem Size')
    plt.ylabel('Value')
    plt.title('Linear Extrapolation with Error Propagation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save and show plot
    plt.tight_layout()
    #plt.savefig('extrapolation_results_linear.pdf', bbox_inches='tight')
    plt.show()