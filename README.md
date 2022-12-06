# ActiveModelH_Heuns
This program is to solve the Active model-H using Heun algorithm.



To compile the code, it requires Cmake, Armadillo linear algebra libraries and MKL libraries. After downloading both the libraries, the code can be compiled
using the command

```
$ mkdir build
$ cd build
$ cmake -S ../ -B .
```

to make the executable file (in the same terminal)
```
$ cd build
$ make
```
Now the executable file **activeH_heun.exe** will be created. To run the program
```
$ ./activeH_heun
```
