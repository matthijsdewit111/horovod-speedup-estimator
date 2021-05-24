To generate benchmark results for the horovod allreduce operation it is recommended to run:

```shell
./run_ar_benchmark [SYSTEM_NAME]
```

For example:

```shell
./run_ar_benchmark lisa
```

Currently supported: (only) `lisa`.

It's also possible to run `allreduce_timing.py` yourself:
```shell
mpirun -np [NUM_MPI_PROCS] python allreduce_timing.py [SYSTEM_NAME]
```

SYSTEM_NAME determines the folder the results are stored in.