import matplotlib.pyplot as plt
import numpy as np

num_procs = np.arange(1, 33)

results = []
for num_proc in num_procs:
    print(num_proc)
    result = np.load(f"real_data/ch4_time_result_{num_proc}.npy") / num_proc
    results.append(result)

results = np.array(results)
# plt.plot(results.T[1])
# plt.show()

seq_results = results[0]
dist_results = results[1:]

speedup = seq_results / dist_results

speedup[speedup < 1] = np.nan

plt.imshow(speedup, cmap='plasma', origin='lower', aspect='auto',
           extent=(1 - .5, 400 + .5, 2 - .5, max(num_procs) + .5))
plt.colorbar(label='speedup')
plt.xlabel('batch size')
plt.ylabel('#MPI processes')
plt.title('Real Speedup CH4')
# plt.xticks(np.arange(20, 40 + 1, 5))
# plt.yticks(np.arange(min(y_values), max(y_values) + 1, 4))
plt.grid(linewidth=0.3)
plt.tight_layout()
plt.savefig('ch4_real.png')
plt.show()
