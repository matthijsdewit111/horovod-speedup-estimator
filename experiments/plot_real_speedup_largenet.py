from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

num_procs = np.arange(1, 33)

results = []
for num_proc in num_procs:
    result = np.load(f"real_data/large_net_time_result_{num_proc}.npy") / num_proc
    results.append(result)

results = np.array(results)

seq_results = results[0]
dist_results = results[1:]

speedup = seq_results / dist_results

speedup[speedup < 1] = np.nan

cmap = copy(cm.get_cmap('plasma'))
cmap.set_bad(color='black')

plt.imshow(speedup, cmap=cmap, origin='lower', aspect='auto',
           extent=(1 - .5, 400 + .5, 2 - .5, max(num_procs) + .5))
plt.colorbar(label='speedup')
plt.xlabel('batch size')
plt.ylabel('#MPI processes')
plt.title('Real Speedup Large FCN')
# plt.xticks(np.arange(min(x_values), max(x_values) + 1, 5))
# plt.yticks(np.arange(min(y_values), max(y_values) + 1, 4))
plt.grid(linewidth=0.3)
plt.tight_layout()
plt.savefig('large_net_real.png')
plt.show()
