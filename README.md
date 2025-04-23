# build
```
mkdir build
cd build
cmake ..
make
```

# test cases
## compile
```
make
```
## export lib paths:
#### Rtune
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../../src/build
```
#### OpenMP offloading
```
export LD_LIBRARY_PATH=/opt/llvm/llvm-14.x-install/lib:$LD_LIBRARY_PATH
```
## run:
```
make run
```
