# build .so
```
mkdir build
cd build
cmake ..
make
```

# examles/jacobi, /lulesh, /stencil
- Use `make` to build binary
- Export library paths
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../src/build
export LD_LIBRARY_PATH=/opt/llvm/llvm-14.x-install/lib:$LD_LIBRARY_PATH
```
- Use `make run` to execute

# examles/LULESH-Rtune
- Insert surrogate modeling, line 2829 - 2850
- Use `make -f Makefile-rtune` to build lulesh

# examles/Castro
- Insert variable tracking, `/Source/driver/main.cpp`, line 273 - 372
- Insert surrogate modeling, `/Source/driver/main.cpp`, line 334 - 448
- Following the instruction in this page: https://amrex-astro.github.io/Castro/docs/getting_started.html to build wdmerger in `/Exec/science/wdmerger`
# examles/impactX-project
- Insert surrogate modeling, `run_surrogate_model.py`, line 106 - 128, line 215 - 265
- Following the instruction in this page: https://impactx.readthedocs.io/en/latest/usage/examples/pytorch_surrogate_model/README.html to run python script
